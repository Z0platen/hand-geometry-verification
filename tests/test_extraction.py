import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from extract_features import archive_index, extract, extract_record, read_metadata


class ExtractionTests(unittest.TestCase):
    def test_duplicate_headers_and_conflicting_image_labels_are_rejected(self):
        malformed = (
            'id,imageName,aspectOfHand,id\n007,hand.jpg,palmar right,008\n',
            'id,imageName,aspectOfHand\n007,hand.jpg,palmar right,extra\n',
            'id,imageName,aspectOfHand\n007,hand.jpg,palmar rite\n',
            'id,imageName,aspectOfHand\n007,hand.jpg,palmar right\n008,hand.jpg,dorsal right\n',
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            for contents in malformed:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        read_metadata(path)

    def test_incomplete_metadata_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            path.write_text('id,imageName,aspectOfHand\n007,hand.jpg\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_metadata(path)

    def test_view_filter_preserves_subject_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            with path.open('w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerows([
                    ['id', 'imageName', 'aspectOfHand'],
                    ['007', 'hand2.jpg', 'palmar right'],
                    ['008', 'hand1.jpg', 'dorsal right'],
                ])
            self.assertEqual(read_metadata(path), [('hand2.jpg', '007')])

    def test_ambiguous_archive_filenames_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photos.zip'
            with ZipFile(path, 'w') as archive:
                archive.writestr('one/photo.jpg', b'a')
                archive.writestr('two/photo.jpg', b'b')
            with ZipFile(path) as archive, self.assertRaises(ValueError):
                archive_index(archive)

    def test_missing_and_corrupt_images_are_retained_with_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photos.zip'
            with ZipFile(path, 'w') as archive:
                archive.writestr('photo.jpg', b'not a jpeg')
                archive.writestr('empty.jpg', b'')
            with ZipFile(path) as archive:
                index = archive_index(archive)
                missing = extract_record(archive, index, 'absent.jpg', '1', None)
                corrupt = extract_record(archive, index, 'photo.jpg', '1', None)
                empty = extract_record(archive, index, 'empty.jpg', '1', None)
            self.assertEqual(missing['status'], 'missing_image')
            self.assertEqual(corrupt['status'], 'unreadable_image')
            self.assertEqual(empty['status'], 'unreadable_image')
            self.assertEqual(len(corrupt['image_sha256']), 64)

    def test_source_paths_cannot_be_overwritten(self):
        path = Path('metadata.csv')
        with self.assertRaises(ValueError):
            extract(Path('photos.zip'), path, Path('model.task'), path)

    def _extraction_inputs(self, folder):
        folder = Path(folder)
        archive, metadata, model = folder / 'photos.zip', folder / 'labels.csv', folder / 'model.task'
        with ZipFile(archive, 'w') as images:
            images.writestr('hand.jpg', b'unreadable image')
        metadata.write_text('id,imageName,aspectOfHand\n007,hand.jpg,palmar right\n', encoding='utf-8')
        model.write_bytes(b'landmarker fixture')
        output = folder / 'features.csv'
        return archive, metadata, model, output

    def test_extraction_publishes_matching_csv_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self._extraction_inputs(folder)
            with patch('extract_features.mp.tasks.vision.HandLandmarker.create_from_options'):
                summary = extract(*inputs)
            output = inputs[-1]
            self.assertEqual(summary['statuses'], {'unreadable_image': 1})
            self.assertEqual(summary['features_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.with_suffix('.meta.json').read_text()), summary)
            self.assertEqual(set(Path(folder).iterdir()), set(inputs) | {output.with_suffix('.meta.json')})

    def test_provenance_write_failure_preserves_existing_output_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self._extraction_inputs(folder)
            output = inputs[-1]
            companion = output.with_suffix('.meta.json')
            output.write_bytes(b'previous csv')
            companion.write_bytes(b'previous provenance')
            with patch('extract_features.mp.tasks.vision.HandLandmarker.create_from_options'), \
                    patch('extract_features.json.dump', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    extract(*inputs)
            self.assertEqual(output.read_bytes(), b'previous csv')
            self.assertEqual(companion.read_bytes(), b'previous provenance')
            self.assertFalse(list(Path(folder).glob('.*-build-*')))

    def test_second_publish_failure_restores_existing_output_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = self._extraction_inputs(folder)
            output = inputs[-1]
            companion = output.with_suffix('.meta.json')
            output.write_bytes(b'previous csv')
            companion.write_bytes(b'previous provenance')
            replace = os.replace

            def fail_companion(source, target):
                if Path(target) == companion:
                    raise OSError('companion unavailable')
                replace(source, target)

            with patch('extract_features.mp.tasks.vision.HandLandmarker.create_from_options'), \
                    patch('extract_features.os.replace', side_effect=fail_companion):
                with self.assertRaises(OSError):
                    extract(*inputs)
            self.assertEqual(output.read_bytes(), b'previous csv')
            self.assertEqual(companion.read_bytes(), b'previous provenance')
            self.assertFalse(list(Path(folder).glob('.*-build-*')))


if __name__ == '__main__':
    unittest.main()
