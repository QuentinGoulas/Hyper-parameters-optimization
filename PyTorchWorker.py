try:
    import torch as th
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
    from torchsummary import summary
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB

import time
import numpy as np

import argparse
import os

import logging
logging.basicConfig(level=logging.DEBUG)

from model import LeNet5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale(num_output_channels=1)
])

class PyTorchWorker(Worker):
        def __init__(self, sleep_interval=0, **kwargs):
                super().__init__(**kwargs)

                self.sleep_interval = sleep_interval
                batch_size = 64

                # This download the train and test data in CIFAR-10. They are not in the repositorie while they are stored in the folder data in the root
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
                self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=2)

                testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
                self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2)


        def compute(self, config, budget, working_directory, *args, **kwargs):
                """
                """

                epochs = budget
                verbose = 'partial'
                device = 'cuda'

                model = LeNet5(config).to(device)

                criterion = th.nn.CrossEntropyLoss()
                optimizer = th.optim.Adam(model.parameters(), lr=0.001)

                model.train()
                torch.backends.cudnn.benchmark = True

                total_time = 0

                for epoch in range(int(np.fix(epochs))):
                        running_loss = 0.0
                        start = time.time()
                        for inputs, labels in self.trainloader:
                                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                                
                                optimizer.zero_grad(set_to_none=True)
                                with th.amp.autocast(device_type='cuda', dtype=torch.float16):
                                        outputs = model(inputs)
                                        loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                                running_loss += loss.item()

                        epoch_time = time.time() - start
                        total_time += epoch_time

                        if verbose=='on' or (verbose=='partial' and ((epoch+1)%5==0 or epoch==0)):
                                print(f"Epoch {epoch+1}/{epochs}, "
                                        f"Loss: {running_loss/len(self.trainloader):.4f}, "
                                        f"Time: {epoch_time:.2f}s, "
                                        f"Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                        for inputs, labels in self.testloader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                outputs = model(inputs)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                print(f"Accuracy: {100 * correct / total:.2f}%")

                return ({
                        'loss': 1-correct/total, # remember: HpBandSter always minimizes!
                        'info': {       
                                'validation accuracy': correct/total,
                                'number of parameters': model.number_of_parameters(),
                                }

                        })

        def evaluate_accuracy(self, model, data_loader):
            model.eval()
            correct=0
            with torch.no_grad():
                    for x, y in data_loader:
                            output = model(x)
                            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                            correct += pred.eq(y.view_as(pred)).sum().item()
            #import pdb; pdb.set_trace()
            accuracy = correct/len(data_loader.sampler)
            return(accuracy)


        @staticmethod
        def get_configspace():
                """
                It builds the configuration space with the needed hyperparameters.
                It is easily possible to implement different types of hyperparameters.
                Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
                :return: ConfigurationsSpace-Object
                """
                cs = CS.ConfigurationSpace()

                #     lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

                # For demonstration purposes, we add different optimizers as categorical hyperparameters.
                # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
                # SGD has a different parameter 'momentum'.
                #     optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

                #     sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

                #     cs.add([lr, optimizer, sgd_momentum])

                # The hyperparameter sgd_momentum will be used,if the configuration
                # contains 'SGD' as optimizer.
                #     cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
                #     cs.add(cond)

                num_dense =  CSH.UniformIntegerHyperparameter('F6', lower=125, upper=325, default_value=125)

                num_filters_1 = CSH.UniformIntegerHyperparameter('C1_chan', lower=1, upper=32, default_value=6, log=True)
                num_filters_2 = CSH.UniformIntegerHyperparameter('C3_chan', lower=1, upper=32, default_value=16, log=True)
                num_filters_3 = CSH.UniformIntegerHyperparameter('C5_chan', lower=40, upper=130, default_value=120, log=True)


                cs.add([num_dense, num_filters_1, num_filters_2, num_filters_3])

                # You can also use inequality conditions:
                #     cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
                #     cs.add(cond)

                #     cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
                #     cs.add(cond)


                #     dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
                #     num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

                #     cs.add([dropout_rate, num_fc_units])

                return cs

'''
We pull off a test script from the HPBandSter documentation

This will most likely be very close to the script implemented in the HyperParameterOptimizer object
'''
if __name__=='__main__':
        min_budget = 1
        max_budget = 4
        n_iterations = 1
        n_workers = 2

        parser = argparse.ArgumentParser(description='Example 3 - Local and Parallel Execution.')
        parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
        args=parser.parse_args()

        # Step 1: Start a nameserver
        # Every run needs a nameserver. It could be a 'static' server with a
        # permanent address, but here it will be started for the local machine with the default port.
        # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
        # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
        NS = hpns.NameServer(run_id='example', host='127.0.0.1', port=None)
        NS.start()
        print('Nameserver started')

        # Step 2: Initialize the workers if calling a subprocess or launch the subprocesses
        if args.worker:
                w = PyTorchWorker(sleep_interval = 0.5, nameserver='127.0.0.1',run_id='example')
                w.run(background=False)
                exit(0)
                print('Workers initialized')
               
        # Step 3: Run an optimizer
        # Now we can create an optimizer object and start the run.
        # We add the min_n_workers argument to the run methods to make the optimizer wait
        # for all workers to start. This is not mandatory, and workers can be added
        # at any time, but if the timing of the run is essential, this can be used to
        # synchronize all workers right at the start.
        bohb = BOHB(configspace = PyTorchWorker.get_configspace(),
                run_id = 'example',
                min_budget=min_budget, max_budget=max_budget
                )
        print('BOHB optimizer initialized')
        res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)
        print('BOHB optimization finished')

        # Step 4: Shutdown
        # After the optimizer run, we must shutdown the master and the nameserver.
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        print('Optimizer object shutdown')

        # Step 5: Analysis
        # Each optimizer returns a hpbandster.core.result.Result object.
        # It holds informations about the optimization run like the incumbent (=best) configuration.
        # For further details about the Result object, see its documentation.
        # Here we simply print out the best config and some statistics about the performed runs.
        print('Analyzing results ...')
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        print('Printing results :')

        all_runs = res.get_all_runs()

        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
        print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
        print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))