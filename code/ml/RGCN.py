import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


preflix = '/Users/willweng/Desktop/research work/casptone/PrimeKG'
out = preflix + '/output/RGCN'
create_inverse_triples = False
training_factory = TriplesFactory.from_path(
        path=f'{out}/train.edgelist',
        create_inverse_triples=create_inverse_triples,
    )
validation_factory = TriplesFactory.from_path(
    path=f'{out}/validation.edgelist',
    create_inverse_triples=create_inverse_triples,
)
testing_factory = TriplesFactory.from_path(
    path=f'{out}/test.edgelist',
    create_inverse_triples=create_inverse_triples,
)

import multiprocessing

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
print(f'the number of cpu cores is {cores} in this computer')


result = pipeline(
    model='RGCN',
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    device="gpu",
    loss='CrossEntropyLoss', # node classification
    model_kwargs= dict(
        decomposition='bases',
        decomposition_kwargs=dict(
        num_bases=3,
        ),
        embedding_dim=64,
        interaction='DistMult',
        num_layers=2,
     ),
     training_kwargs=dict(
         num_epochs=150,
         sampler='schlichtkrull',
         batch_size=1024, # larger, the quicker, cost increases as O(n^2), time decreases as O(n)
         num_workers=12,
     ),
     training_loop="slcwa",
     regularizer = "no",
     optimizer="Adam",
     optimizer_kwargs=dict(
         lr=0.005,
     ),
     negative_sampler="basic",
     evaluator='RankBasedEvaluator',
     evaluator_kwargs=dict(
         filtered=True,
     ),
     evaluation_kwargs=dict(batch_size=512),  # Batch size for evaluation
     stopper='early',
     stopper_kwargs=dict(frequency=25,patience=4,relative_delta=0.002),
     use_tqdm=True,
     random_seed=42
)

result.save_to_directory(preflix + '/output/RGCN/PrimekG2')