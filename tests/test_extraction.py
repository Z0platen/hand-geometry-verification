import csv
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from extract_features import archive_index, extract, extract_record, read_metadata


class ExtractionTests(unittest.TestCase):
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
            with ZipFile(path) as archive:
                index = archive_index(archive)
                missing = extract_record(archive, index, 'absent.jpg', '1', None)
                corrupt = extract_record(archive, index, 'photo.jpg', '1', None)
            self.assertEqual(missing['status'], 'missing_image')
            self.assertEqual(corrupt['status'], 'unreadable_image')
            self.assertEqual(len(corrupt['image_sha256']), 64)

    def test_source_paths_cannot_be_overwritten(self):
        path = Path('metadata.csv')
        with self.assertRaises(ValueError):
            extract(Path('photos.zip'), path, Path('model.task'), path)


if __name__ == '__main__':
    unittest.main()
