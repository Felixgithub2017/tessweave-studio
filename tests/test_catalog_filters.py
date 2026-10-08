import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.hub import filter_catalog, catalog
from workbench.inspection import create_directory


class CatalogFilters(unittest.TestCase):
    def test_range_dates_modality(self):
        rows=[{'id':'a','parameters':7e9,'created_at':'2026-09-01T00:00:00Z','pipeline_tag':'text-generation'},
              {'id':'b','parameters':None,'created_at':None}]
        self.assertEqual(filter_catalog(rows,{'min_b':6,'max_b':8,'after':'2026-08-01','before':'2026-10-01','modality':'text-generation'}),rows[:1])
        self.assertEqual(filter_catalog(rows,{}),rows)
        for f in ({'min_b':9,'max_b':2},{'min_b':float('nan')},{'after':'bad'}):
            with self.assertRaises(ValueError):filter_catalog(rows,f)

    def test_hf_metadata_not_name(self):
        with patch('workbench.hub.get_json',return_value=[{'id':'Qwen/777B','createdAt':'2026-09-01','safetensors':{'total':7e9}}]):
            r=catalog('huggingface',filters={'min_b':1})
            self.assertEqual(r['models'],[]) # only integer counts accepted
        with patch('workbench.hub.get_json',return_value=[{'id':'Qwen/777B','safetensors':{'total':7000000000}}]):
            self.assertEqual(catalog('huggingface',filters={'max_b':8})['models'][0]['parameters'],7000000000)

    def test_create_folder_no_overwrite_or_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve()
            r=create_directory(str(base),'模型库',[base]);self.assertTrue(Path(r['path']).is_dir())
            with self.assertRaises(FileExistsError):create_directory(str(base),'模型库',[base])
            for name in ('../escape','a/b','..',''):
                with self.assertRaises(ValueError):create_directory(str(base),name,[base])
