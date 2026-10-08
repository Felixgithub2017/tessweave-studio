import unittest
from unittest.mock import patch
from workbench.environment_inventory import assess, scan, inventory


class EnvironmentInventoryTests(unittest.TestCase):
    def test_conda_default_plan_is_explicit_and_does_not_create(self):
        import tempfile
        from pathlib import Path
        from workbench.environments import environment_plan
        with tempfile.TemporaryDirectory() as root:
            conda=Path(root)/'conda';conda.touch();conda.chmod(0o700)
            plan=environment_plan({'path':root+'/new-env','conda':str(conda),'recipe':'download'},[root],root)
            self.assertEqual(plan['request']['manager'],'conda')
            self.assertEqual(plan['steps'][0][:3],[str(conda.resolve()),'create','--yes'])
            self.assertIn('python=3.11',plan['steps'][0])
            self.assertIn('--override-channels',plan['steps'][0])
            self.assertFalse((Path(root)/'new-env').exists())
            with self.assertRaises(ValueError):
                environment_plan({'path':root+'/another','conda':str(conda),'python_version':'3.11;id'},[root],root)

    def test_missing_conda_does_not_silently_fallback(self):
        import tempfile
        from workbench.environments import environment_plan
        with tempfile.TemporaryDirectory() as root, patch('workbench.environments.conda_installations',return_value=[]):
            with self.assertRaisesRegex(ValueError,'not found'):
                environment_plan({'path':root+'/env'},[root],root)

    def test_mlx_core_is_not_mlx_lm(self):
        report={'os':'Darwin','machine':'arm64','environments':[{'returncode':0,'packages':{'mlx':'0.31.2','mflux':'0.18.1'}}]}
        row=assess(report,'mlx')['environments'][0]
        self.assertFalse(row['candidate'])
        self.assertIn('Missing mlx-lm',row['reasons'])

    def test_blank_environment_path_has_actionable_error(self):
        from workbench.environments import environment_plan
        with self.assertRaisesRegex(ValueError,'absolute target folder'):
            environment_plan({'path':'','recipe':'mlx'},[], '/tmp')

    def test_missing_packages_and_host_are_not_candidates(self):
        report={'os':'Linux','machine':'x86_64','environments':[{'returncode':0,'packages':{'mlx-lm':'1.0'}}]}
        self.assertFalse(assess(report,'mlx')['environments'][0]['candidate'])

    def test_multiple_candidates_preserved(self):
        report={'os':'Darwin','machine':'arm64','environments':[
            {'path':p,'returncode':0,'packages':{'mlx-lm':'1.0'}} for p in ['/a/bin/python','/b/bin/python']]}
        self.assertTrue(all(x['candidate'] for x in assess(report,'mlx')['environments']))

    def test_ssh_validation_before_execution(self):
        with patch('workbench.environment_inventory.subprocess.run') as run:
            for alias in ['-oX','host;id','user@host']:
                with self.assertRaises(ValueError):scan(alias)
            run.assert_not_called()

    def test_real_python_metadata_probe_without_install(self):
        report=inventory()
        self.assertTrue(report['environments'])
        self.assertTrue(any(r.get('version') for r in report['environments']))
        self.assertTrue(all('-I' in r['command'] for r in report['environments']))

    def test_swift_version_contract(self):
        report={'os':'Linux','machine':'x86_64','environments':[{'returncode':0,'packages':{'ms-swift':'5.0','torch':'2.9'}}]}
        self.assertFalse(assess(report,'train')['environments'][0]['candidate'])
