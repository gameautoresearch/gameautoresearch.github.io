"""Adapter-local checkpoint enrichment with one atomic native serialization.

The native checkpoint method remains the sole owner of its complete payload and
state schema. This object forwards its Torch operations and supplies only the
save callback for one Trainer.checkpoint invocation; global torch.save is never
patched. Extra metadata cannot replace any native field.
"""
from pathlib import Path
import errno,os

class CheckpointWriter:
    def __init__(self,torch,metadata):
        if not isinstance(metadata,dict) or set(metadata)!={'entrant_binding','entrant_state'}:
            raise ValueError('exact entrant checkpoint metadata required')
        self.torch=torch;self.metadata=metadata;self.calls=0;self.file_fsynced=False
    def __getattr__(self,name):
        return getattr(self.torch,name)
    def save(self,payload,path):
        if self.calls or set(payload)&set(self.metadata):
            raise ValueError('checkpoint metadata collision or repeated serialization')
        self.calls+=1
        # The native method writes a sibling temporary file and publishes it by
        # atomic rename only after this callback returns. Flush it before return.
        enriched={**payload,**self.metadata}
        with Path(path).open('wb') as stream:
            self.torch.save(enriched,stream)
            stream.flush();os.fsync(stream.fileno());self.file_fsynced=True

def checkpoint(trainer,path,elapsed,*,entrant_binding,entrant_state):
    """Call the unchanged native checkpoint writer with an instance-local hook."""
    from simulators.clash.native.train import Trainer
    path=Path(path);temporary=path.with_suffix('.tmp');original=trainer.torch
    writer=CheckpointWriter(original,{'entrant_binding':entrant_binding,'entrant_state':entrant_state})
    trainer.torch=writer
    try:
        diagnostic=Trainer.checkpoint(trainer,path,elapsed)
        if writer.calls!=1 or not writer.file_fsynced:raise ValueError('native checkpoint did not serialize exactly once')
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:
            try:os.fsync(descriptor);directory_fsynced=True
            except OSError as exc:
                if exc.errno not in (errno.EINVAL,errno.ENOTSUP):raise
                # Some volume filesystems do not implement directory fsync.
                # Worker sync/volume publication still precedes provider shutdown.
                directory_fsynced=False
        finally:os.close(descriptor)
        diagnostic.update(serialization_passes=1,file_fsynced=True,directory_fsynced=directory_fsynced,metadata_attached_before_serialization=True)
        return diagnostic
    finally:
        trainer.torch=original
        if temporary.exists():temporary.unlink()
