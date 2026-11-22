"""The researcher rulebook boundary is Markdown-only, including exported copies."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('research_tasks', Path(__file__).parents[1] / 'research_tasks.py')
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class FlexibleRulebookTest(unittest.TestCase):
    def make_fixture(self, root):
        import json
        entry = {'autoresearch': 'simulators/example/autoresearch'}
        (root/'simulators').mkdir()
        (root/'simulators/catalog.json').write_text(json.dumps({'games': {'example': entry}}))
        task = root/entry['autoresearch']/'task'
        rules = task/'game_rules'
        (rules/'entities').mkdir(parents=True)
        (rules/'README.md').write_text('# Mode\n\n[Creatures](entities/creatures.md)\n')
        (rules/'entities/creatures.md').write_text('# Creatures\n\n[Rules](../README.md)\n\n| Species | HP |\n|---|---:|\n| Example | 10 |\n')
        return rules

    def test_nested_game_specific_markdown_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_fixture(root)
            self.assertEqual(tasks.check_rules(root), [])
            files = tasks.rulebook_files('example', root)
            self.assertEqual(files[0], 'README.md')
            self.assertIn('entities/creatures.md', files)
            self.assertTrue(all(name.endswith('.md') for name in files))

    def test_export_contains_only_markdown_and_integrity_stays_with_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_fixture(root)
            out = root/'export'
            result = tasks.export_rulebook('example', out, root)
            self.assertEqual(set(result['files_sha256']), set(tasks.rulebook_files('example', root)))
            for name, expected in result['files_sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), expected)
            self.assertFalse((out/'wiki-manifest.json').exists())
            self.assertTrue(all(p.suffix == '.md' for p in out.rglob('*') if p.is_file()))
            with self.assertRaises(FileExistsError):tasks.export_rulebook('example', out, root)

    def test_scripts_and_metadata_cannot_leak_into_agent_packet(self):
        for name in ['build_wiki.py','coverage.json','raw-source.js']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rules=self.make_fixture(root);(rules/name).write_text('not agent rules')
                with self.assertRaises(ValueError):tasks.rulebook_files('example', root)
                self.assertTrue(tasks.check_rules(root))

    def test_embedded_code_is_rejected_even_in_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rules=self.make_fixture(root)
            (rules/'bad.md').write_text('# Effect\n\n```javascript\nreturn null;\n```\n')
            self.assertTrue(any('code/implementation' in error for error in tasks.check_rules(root)))

    def test_missing_nested_page_and_external_links_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rules=self.make_fixture(root)
            (rules/'entities/creatures.md').unlink()
            self.assertTrue(any('broken wiki link' in e for e in tasks.check_rules(root)))
            (rules/'README.md').write_text('[Source](../../../../../source.py)')
            self.assertTrue(any('escapes' in e for e in tasks.check_rules(root)))

    def test_outside_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rules=self.make_fixture(root)
            (rules/'escape.md').symlink_to(root/'simulators/catalog.json')
            with self.assertRaises(ValueError):tasks.rulebook_files('example', root)

    def test_all_six_live_rulebooks_match_delivery_contract(self):
        self.assertEqual(tasks.check_rules(), [])
        for game, entry in tasks.catalog().items():
            base = tasks.ROOT / entry['autoresearch'] / 'task/game_rules'
            names = tasks.rulebook_files(game)
            self.assertTrue(names)
            for name in names:
                text = (base / name).read_text()
                self.assertTrue(name.endswith('.md'), (game, name))
                self.assertNotIn('```', text, (game, name))
                self.assertNotIn('Exact event callbacks', text, (game, name))


if __name__ == '__main__':unittest.main()
