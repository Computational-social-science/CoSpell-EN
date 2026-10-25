"""Tests for release integrity and lossless, non-overwriting preparation."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_workspace', ROOT/'prepare_workspace.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_all_four_parts_of_speech_and_lossless_round_trip(self):
        manifest, profiles, contents = release.read_release(ROOT)
        self.assertEqual(set(profiles), {'all','noun','verb','adjective','adverb'})
        self.assertEqual(len(contents), 51)
        self.assertEqual(len(profiles['all']['files']), 86)
        with tempfile.TemporaryDirectory() as temporary:
            for profile,info in profiles.items():
                target = Path(temporary)/profile
                release.prepare(profile, target, ROOT)
                for name,source in info['files'].items():
                    self.assertEqual((target/name).read_bytes(), contents[source])

    def test_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary)/'existing.txt'
            sentinel.write_text('preserve',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                release.prepare('all',Path(temporary),ROOT)
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertEqual(len(list(Path(temporary).iterdir())),1)

    def test_rejects_unsafe_manifest_paths(self):
        for name in ('../escape','/absolute','C:/outside','a/../../outside','a\\outside',''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                release.inside(ROOT,name)

    def test_detects_modified_source_before_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config').mkdir()
            (root/'changed.py').write_text('x=2\n',encoding='utf-8')
            manifest = {'format_version':1,'files':[{
                'path':'changed.py','sha256':'0'*64,'original_locations':['old.py']} ]}
            (root/'config/source_map.json').write_text(json.dumps(manifest),encoding='utf-8')
            (root/'config/profiles.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                release.read_release(root)


if __name__ == '__main__':
    unittest.main()
