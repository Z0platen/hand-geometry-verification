import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate import error_curve, evaluate, read_features, summarize_scores
from matching import FEATURE_SETS


FEATURE_COLUMNS = [column for columns in FEATURE_SETS.values() for column in columns]
COLUMNS = ['image_name', 'subject_id', 'image_sha256', 'status', *FEATURE_COLUMNS]


def sample(subject, number):
    name = f'{subject}_{number}.jpg'
    row = {
        'image_name': name, 'subject_id': subject, 'status': 'ok',
        'image_sha256': hashlib.sha256(name.encode()).hexdigest(),
    }
    offset = 0 if subject == 'a' else 100
    for column in FEATURE_COLUMNS:
        row[column] = offset + number + 1
    return row


def write_features(path, rows, columns=COLUMNS):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class EvaluationTests(unittest.TestCase):
    def test_malformed_csv_headers_and_row_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            row = sample('a', 0)
            for columns in (COLUMNS + ['subject_id'], COLUMNS + ['']):
                with self.subTest(columns=columns):
                    write_features(path, [row], columns)
                    with self.assertRaises(ValueError):
                        read_features(path)
            write_features(path, [row])
            valid = path.read_text()
            for ending in (',unexpected\n', '\n'):
                with self.subTest(ending=ending):
                    header, values = valid.splitlines()
                    if ending == '\n':
                        values = values.rsplit(',', 1)[0]
                    path.write_text(header + '\n' + values + ending)
                    with self.assertRaises(ValueError):
                        read_features(path)

    def test_unknown_status_and_invalid_successful_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            for change in ({'status': 'okay'}, {'thumb_px': 'nan'}, {'thumb_ratio': '0'}):
                with self.subTest(change=change):
                    write_features(path, [{**sample('a', 0), **change}])
                    with self.assertRaises(ValueError):
                        read_features(path)

    def test_relabelled_duplicate_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            first = sample('a', 0)
            second = {**sample('b', 0), 'image_name': first['image_name']}
            write_features(path, [first, second])
            with self.assertRaisesRegex(ValueError, 'Duplicate image name'):
                read_features(path)

    def test_trimmed_labels_are_consistent_and_failure_subjects_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            valid = {**sample('a', 0), 'subject_id': ' a ', 'status': ' ok '}
            failure = dict.fromkeys(COLUMNS, '')
            failure.update(image_name='failed.jpg', subject_id='failed', status='no_hand')
            write_features(path, [valid, failure])
            rows, statuses, subjects = read_features(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['subject_id'], rows[0]['status']), ('a', 'ok'))
            self.assertEqual(statuses, {'ok': 1, 'no_hand': 1})
            self.assertEqual(subjects, {'a': 1, 'failed': 1})

    def test_scores_require_matching_shapes_and_boolean_labels(self):
        invalid = (
            ([0.1, 0.2], [True, False, False]),
            ([[0.1, 0.2]], [True, False]),
            ([0.1, 0.2], ['True', 'False']),
            ([0.1, 0.2], [1, 0]),
            ([-0.1, 0.2], [True, False]),
        )
        for distances, labels in invalid:
            with self.subTest(distances=distances, labels=labels):
                with self.assertRaises(ValueError):
                    error_curve(distances, labels)
                with self.assertRaises(ValueError):
                    summarize_scores(distances, labels, 0.15)
        for threshold in (float('nan'), float('inf'), True):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    summarize_scores([0.1, 0.2], [True, False], threshold)

    def test_valid_list_and_matrix_scores_agree(self):
        flat = summarize_scores([0.1, 0.2], [True, False], 0.15)
        matrix = summarize_scores(np.array([[0.1, 0.2]]), np.array([[True, False]]), 0.15)
        self.assertEqual(flat, matrix)
        self.assertEqual(flat['fmr'], 0)
        self.assertEqual(flat['fnmr'], 0)

    def test_evaluation_failure_publishes_no_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path, output = Path(folder) / 'features.csv', Path(folder) / 'evaluation'
            write_features(path, [sample(subject, i) for subject in ('a', 'b') for i in range(6)])
            output.mkdir()
            with patch('evaluate.plot_results', side_effect=OSError('plot write failed')):
                with self.assertRaises(OSError):
                    evaluate(path, output)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertFalse(list(Path(folder).glob('evaluation-build-*')))

    def test_existing_output_and_source_file_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path, output = Path(folder) / 'features.csv', Path(folder) / 'evaluation'
            write_features(path, [sample(subject, i) for subject in ('a', 'b') for i in range(6)])
            original = path.read_bytes()
            output.mkdir()
            previous = output / 'previous.txt'
            previous.write_text('previous run')
            for target in (path, output):
                with self.subTest(target=target):
                    with self.assertRaises(ValueError):
                        evaluate(path, target)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(previous.read_text(), 'previous run')


if __name__ == '__main__':
    unittest.main()
