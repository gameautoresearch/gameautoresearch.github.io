//! Worker threads that run games (the batched environment's pool).
//!
//! Combats and effect chains on huge late warbands go deeper than a thread's default 2 MiB
//! stack: inside the Python extension that is a bare SIGSEGV with an empty log. Only the
//! pages a thread touches are committed, so the larger reserve costs no memory.

/// Stack of a game worker thread.
pub const STACK: usize = 64 << 20;

/// `s.spawn(f)` with the worker stack.
pub fn spawn<'scope, 'env, T: Send + 'scope>(
    s: &'scope std::thread::Scope<'scope, 'env>,
    f: impl FnOnce() -> T + Send + 'scope,
) -> std::thread::ScopedJoinHandle<'scope, T> {
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn_scoped(s, move || {
            crate::guard::thread_started(STACK);
            f()
        })
        .expect("spawn a game worker thread")
}

/// Reuse the most recently requested pool; workers retain scoped-engine stack guards.
/// Active environments own Arcs. Evicting an idle size drops its threads rather than
/// retaining one full pool forever for every size used by an interactive researcher.
pub fn pool(threads: usize) -> std::sync::Arc<rayon::ThreadPool> {
    use std::sync::{Arc, Mutex, OnceLock};
    type CachedPool = Option<(usize, Arc<rayon::ThreadPool>)>;
    static POOL: OnceLock<Mutex<CachedPool>> = OnceLock::new();
    let mut cached = POOL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap();
    if let Some((size, pool)) = &*cached {
        if *size == threads {
            return pool.clone();
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(STACK)
            .start_handler(|_| crate::guard::thread_started(STACK))
            .thread_name(|i| format!("tavern-persistent-{i}"))
            .build()
            .expect("persistent game worker pool"),
    );
    *cached = Some((threads, pool.clone()));
    pool
}
