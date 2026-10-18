import importlib.util
from pathlib import Path
import shutil
import unittest
import uuid
from llm_kt.data import import_snapshot

spec=importlib.util.spec_from_file_location('prior_builder',Path(__file__).parents[1]/'tools/build_prompt_priors.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class PriorTests(unittest.TestCase):
    def test_validation_label_changes_do_not_change_training_priors(self):
        root=Path(__file__).parent/('work_'+uuid.uuid4().hex);root.mkdir()
        try:
            results=[]
            for version,validation_answers in enumerate(['1,1,1','0,0,0']):
                train=root/f'train{version}.csv'
                train.write_text('fold,uid,questions,concepts,responses\n'
                                 f'0,validation_student,"1,2,3","1,2,3","{validation_answers}"\n'
                                 '1,training_student,"1,2,3","1,2,3","1,0,1"\n',encoding='utf-8')
                snapshot=root/f'snapshot{version}'
                import_snapshot(Path(__file__).parent/'fixtures/test_quelevel.csv',snapshot,'synthetic',train_source=train)
                results.append(builder.build(snapshot,root/f'prior{version}'))
            for field in ['global','questions','concepts']:
                self.assertEqual(results[0][field],results[1][field])
            self.assertEqual(results[0]['global']['n'],3)
            self.assertEqual(results[0]['n_training_students'],1)
        finally:
            if root.resolve().parent!=Path(__file__).parent.resolve() or not root.name.startswith('work_'):
                raise RuntimeError('Unsafe cleanup path')
            shutil.rmtree(root)
