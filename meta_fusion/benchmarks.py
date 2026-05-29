import pdb
import itertools
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
import random
import time
import os
import copy
from tqdm import tqdm

sys.path.append('../')
from meta_fusion.utils import *
from meta_fusion.methods import Trainer

class Benchmarks(Trainer):
    """
    Implements benchmark fusion baselines using the original framework style.

    Benchmarks:
      - modality_j: each unimodal model alone
      - early_fusion: model trained on all modalities jointly
      - late_fusion: simple average of unimodal model outputs

    Notes:
      - Keeps the original 'imputation-style' behavior: missing modalities
        are passed through as they appear in the inputs.
      - Updated for classification metrics beyond accuracy.
      - Initialization made consistent with the rest of your Alzheimer code.
    """

    def __init__(self, config, models, data_loaders, model_dims=None):
        super().__init__(config, models, data_loaders)

        # Optional validation if model_dims are provided
        if model_dims is not None:
            self._validate_models(models, model_dims)
        else:
            self.models = models
            self.model_dims = None

        self.rho_list = [0]

        self.model_num = len(self.models)
        assert self.model_num >= 2, "Expecting at least two benchmark models!"

        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.optimizers = []
        for i in range(self.model_num):
            self.models[i].to(self.device)
            optimizer = optim.Adam(
                self.models[i].parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.optimizers.append(optimizer)

        self.initial_model_states = [copy.deepcopy(model.state_dict()) for model in self.models]
        self.initial_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in self.optimizers]

    def _validate_models(self, models, model_dims):
        """
        Optional validation of the benchmark model list.
        Expects:
          - n single-modality models
          - 1 all-modality (early fusion) model
        """
        n = len(model_dims[0])  # number of modalities
        single_modality_models = [None] * n
        all_modalities_model = None

        for model, dims in zip(models, model_dims):
            non_zero_count = sum(dim != 0 for dim in dims)

            if non_zero_count == 1:
                index = dims.index(next(filter(lambda x: x != 0, dims)))
                if single_modality_models[index] is not None:
                    raise ValueError(f"More than one single-modal model is trained on modality {index+1}.")
                single_modality_models[index] = (model, dims)

            elif non_zero_count == n:
                if all_modalities_model is not None:
                    raise ValueError("More than one model is trained on all modalities.")
                all_modalities_model = (model, dims)

        if None in single_modality_models:
            raise ValueError("Some single modality models are missing.")

        if all_modalities_model is None:
            raise ValueError("No model is trained on all modalities.")

        self.models = [model for model, _ in single_modality_models] + [all_modalities_model[0]]
        self.model_dims = [dims for _, dims in single_modality_models] + [all_modalities_model[1]]

    def train(self):
        print("Start training benchmark models...")
        super().train_all_rhos()
        print("Finished training benchmark models!")

    def test(self, test_loader, benchmarks=None):
        valid_benchmarks = ["modality", "early_fusion", "late_fusion"]

        if benchmarks is None:
            benchmarks = valid_benchmarks
        elif not set(benchmarks).issubset(valid_benchmarks):
            raise ValueError("Some benchmarks are invalid.")

        _ = self.load_checkpoints(0)

        if self.task_type == "classification":
            results = self.test_classification(benchmarks, test_loader)
        else:
            results = self.test_regression(benchmarks, test_loader)

        return results

    def test_regression(self, benchmarks, test_loader):
        full_mse = {}
        method_mse = {}

        for i in range(self.model_num):
            self.models[i].eval()

        with torch.no_grad():
            for batch in test_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                outputs = [model(modalities) for model in self.models]

                for benchmark in benchmarks:
                    if benchmark == "modality":
                        for j in range(len(modalities)):
                            key = f"modality_{j+1}"
                            if key not in method_mse:
                                method_mse[key] = AverageMeter()
                            final_output = outputs[j]
                            method_mse[key].update(self.loss_mse(final_output, target), target.size()[0])

                    elif benchmark == "early_fusion":
                        if "early_fusion" not in method_mse:
                            method_mse["early_fusion"] = AverageMeter()
                        final_output = outputs[-1]
                        method_mse["early_fusion"].update(self.loss_mse(final_output, target), target.size()[0])

                    elif benchmark == "late_fusion":
                        if "late_fusion" not in method_mse:
                            method_mse["late_fusion"] = AverageMeter()
                        final_output = torch.mean(torch.stack(outputs[:-1]), dim=0)
                        method_mse["late_fusion"].update(self.loss_mse(final_output, target), target.size()[0])

        for key, meter in method_mse.items():
            mse_value = meter.avg
            full_mse[key] = mse_value.item() if hasattr(mse_value, 'item') else float(mse_value)

        if self.verbose:
            for key, value in full_mse.items():
                print(f"Method: ({key}), Test_MSE: {value}")

        return full_mse

    def test_classification(self, benchmarks, test_loader):
        """
        Returns full multiclass metrics for each benchmark:
          - accuracy
          - macro_f1
          - auc_ovr
          - sensitivity
          - specificity
        """
        records = {}

        def init_record():
            return {"y_true": [], "y_pred": [], "y_prob": []}

        for benchmark in benchmarks:
            if benchmark == "modality":
                # one record per unimodal branch
                # assumes all but the last model are unimodal, last is early fusion
                for j in range(self.model_num - 1):
                    records[f"modality_{j+1}"] = init_record()
            else:
                records[benchmark] = init_record()

        for i in range(self.model_num):
            self.models[i].eval()

        with torch.no_grad():
            for batch in test_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                outputs = [model(modalities) for model in self.models]

                if "modality" in benchmarks:
                    for j in range(len(modalities)):
                        key = f"modality_{j+1}"
                        final_output = outputs[j]
                        prob = torch.softmax(final_output, dim=1).cpu().numpy()
                        pred = torch.argmax(final_output, dim=1).cpu().numpy()
                        records[key]["y_true"].append(target.cpu().numpy())
                        records[key]["y_pred"].append(pred)
                        records[key]["y_prob"].append(prob)

                if "early_fusion" in benchmarks:
                    final_output = outputs[-1]
                    prob = torch.softmax(final_output, dim=1).cpu().numpy()
                    pred = torch.argmax(final_output, dim=1).cpu().numpy()
                    records["early_fusion"]["y_true"].append(target.cpu().numpy())
                    records["early_fusion"]["y_pred"].append(pred)
                    records["early_fusion"]["y_prob"].append(prob)

                if "late_fusion" in benchmarks:
                    final_output = torch.mean(torch.stack(outputs[:-1]), dim=0)
                    prob = torch.softmax(final_output, dim=1).cpu().numpy()
                    pred = torch.argmax(final_output, dim=1).cpu().numpy()
                    records["late_fusion"]["y_true"].append(target.cpu().numpy())
                    records["late_fusion"]["y_pred"].append(pred)
                    records["late_fusion"]["y_prob"].append(prob)

        full_results = {}
        for key, rec in records.items():
            y_true = np.concatenate(rec["y_true"])
            y_pred = np.concatenate(rec["y_pred"])
            y_prob = np.concatenate(rec["y_prob"])
            full_results[key] = compute_classification_metrics(
                y_true, y_pred, y_prob, num_classes=self.num_classes
            )

        if self.verbose:
            for key, value in full_results.items():
                print(f"Method: ({key}), Metrics: {value}")

        return full_results





class Coop(Trainer):
    """
    Implements original oversion of the cooperative learning via alternating gradient descent. 
    Aside from the disagreement penalty, each model is trained to match the partial residual of 
    the ground truth.

    Note that the original paper did not specify how to implement cooperative learning of
    more than two modalities with interactions, hence this class only considers two modalities.
    """
    def __init__(self, config, models, model_dims, data_loaders):
        super().__init__(config, models, data_loaders)

        # Identify model types by checking dimensions
        for i, (dim1, dim2) in enumerate(model_dims):
            if dim1 != 0 and dim2 != 0:
                self.idx_early = i
            elif dim1 == 0:   # only modality 2 is used 
                self.idx_mod2 = i
            elif dim2 == 0:   # only modality 1 is used
                self.idx_mod1 = i
        
        if len(models) == 2:
            self.idx_early = None
        
        # Original cooperative learning only supports regression
        self.task_type = "regression"

        # range of rho should be [0,1)
        self.rho_list = [rho for rho in self.rho_list if 0 <= rho < 1]


    def train(self):
        """
        Main training function that performs the following:
        1. trains benchmark models and a list of rhos provided by the user;
        2. chooses and record the best rho via cross-validation.
        """
        
        print("Start training benchmark models...")
        super().train_all_rhos()
        print("Finished training benchmark models!")

        print("Selecting the optimal disgreement penalty via cross-validation...")
        super().choose_best_rho()
        print("Done!")


    
    def test(self, test_loader):
        """
        Evaluate the cooperative learning performance on the test data 
        """

        # Make sure the best model is loaded
        _ = self.load_checkpoints(self.best_rho)

        results = self.test_regression(test_loader)
        
        return results



    def train_fixed_rho(self, rho):
        """
        Train the models for a fixed level of disagreement penalty. 
        """
        
        # Variable to keep track of the best validation loss
        self.best_val_task_losses = [float('inf')] * self.model_num

        for epoch in range(self.epochs):

            # for scheduler in self.schedulers:
            #     scheduler.step(epoch)
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )
            
                train_losses, train_task_losses = self.train_one_epoch(epoch, rho)
                val_losses, val_task_losses = self.validate(epoch, rho)
                
            for i in range(self.model_num):
                # Check if the current model has the best validation task loss
                is_best = val_task_losses[i].avg < self.best_val_task_losses[i]
                
                # Print train and validation losses
                msg1 = "model_{:d}: train loss: {:.3f}, train task loss: {:.3f} "
                msg2 = "- val loss: {:.3f}, val task loss: {:.3f}"
                if is_best:
                    msg2 += " [*] Best so far"
                msg = msg1 + msg2
                
                if self.verbose:
                    print(
                        msg.format(i+1, train_losses[i].avg, train_task_losses[i].avg, 
                                        val_losses[i].avg, val_task_losses[i].avg)
                    )

                # If this is the best model so far, update best loss and save checkpoint
                if is_best:
                    self.best_val_task_losses[i] = val_task_losses[i].avg
                    self.save_checkpoint(rho, i,
                        {
                            "epoch": epoch + 1,
                            "disagreement_penalty": rho,
                            "model_state": self.models[i].state_dict(),
                            "optim_state": self.optimizers[i].state_dict(),
                            "best_val_task_loss": self.best_val_task_losses[i],
                        }
                    )



    def train_one_epoch(self, epoch, rho):
        """
        Train the model for 1 epoch of the training set.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        task_losses = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())


        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch in self.train_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                outputs = [model(modalities) for model in self.models]
                for i in range(self.model_num):

                    # model fitted on partial residuals
                    if i == self.idx_early:
                        final_output = outputs[self.idx_mod1].detach() + outputs[self.idx_mod2].detach() + outputs[i]
                        divergence_loss = rho / (1 - rho) * self.loss_mse(outputs[i],torch.zeros_like(outputs[i]))
                    
                    elif i == self.idx_mod1:
                        final_output = outputs[self.idx_mod2].detach() + outputs[i] if not self.idx_early else\
                                      outputs[self.idx_mod2].detach() + outputs[self.idx_early].detach() + outputs[i]
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod2].detach())
                    
                    elif i == self.idx_mod2:
                        final_output = outputs[self.idx_mod1].detach() + outputs[i] if not self.idx_early else\
                                      outputs[self.idx_mod1].detach() + outputs[self.idx_early].detach() + outputs[i]
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod1].detach())

                    task_loss = self.loss_mse(final_output, target)
                    loss = task_loss + divergence_loss
                    
                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

                    # compute gradients and update optimizer
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    #self.schedulers[i].step()
                
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                # pbar.set_description(
                #     (
                #         "{:.1f}s - average loss: {:.3f} - average task loss: {:.3f}".format(
                #             (toc-tic), np.mean([losses[i].avg for i in range(self.model_num)]), 
                #                        np.mean([task_losses[i].avg for i in range(self.model_num)])
                #         )
                #     )
                # )
                self.batch_size = target.shape[0]
                pbar.update(self.batch_size)
            
            return losses, task_losses



    def validate(self, epoch, rho):
        """
        Evaluate the model on the validation set on the current model.
        """
        losses = []
        task_losses = []
        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())
        
        with torch.no_grad():
            for batch in self.val_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                outputs = [model(modalities) for model in self.models]            
                final_output = torch.sum(torch.stack(outputs), dim=0)

                for i in range(self.model_num):
                    # model fitted on partial residuals
                    if i == self.idx_early:
                        divergence_loss = rho / (1 - rho) * self.loss_mse(outputs[i],torch.zeros_like(outputs[i]))
                    
                    elif i == self.idx_mod1:
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod2])
                    
                    elif i == self.idx_mod2:
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod1])

                    task_loss = self.loss_mse(final_output, target)
                    loss = task_loss + divergence_loss

                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

        return losses, task_losses


    
    def test_regression(self, test_loader):
        """
        Evaluate the performance of the ensemble on the validation set for regression tasks.
        """

        method_mse = AverageMeter()

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch in test_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                outputs = [model(modalities) for model in self.models]

                final_output = torch.sum(torch.stack(outputs), dim=0)
                method_mse.update(self.loss_mse(final_output, target), target.size()[0])
            
            mse_value = method_mse.avg
            if hasattr(mse_value, 'item'):
                mse_value = mse_value.item()
            else:
                mse_value =  float(mse_value)

            if self.verbose:
                print(f'Method: ({"coop"}), Test_MSE: {mse_value}')

            return {"coop":mse_value}





class AdversarialTrainer(Trainer):
    """
    Modifies the original Meta Fusion framework to showcase the effect of negative learning. 
    Disagreement penalty is fixed, weights of adaptive mutual learning is adversely selected.
    """
    def __init__(self, config, models, data_loaders):
        super().__init__(config, models, data_loaders)
        self.ckpt_dir_root = config["ckpt_dir"]
        self.ckpt_dir_adaptive = os.path.join(self.ckpt_dir_root, "adaptive")
        self.ckpt_dir_adversarial = os.path.join(self.ckpt_dir_root, "adversarial")
        self.fixed_rho = np.max(self.rho_list)

    def train_adaptive(self):
        """
        Training with the normal meta fusion weights
        """
        print("Start training student cohort with normal adaptive weights...")
        self.ckpt_dir = self.ckpt_dir_adaptive   # set the correct checkpoint saving directory
        self.clustering_weights, self.cluster_idxs = None, None  # initialize weights

        for rho in self.rho_list:
            print(f"Training with disagreement penalty = {rho}")

            if rho > 0 and not hasattr(self, 'rho0_losses'):
                self.rho0_losses = self.load_checkpoints(0)

            # Reset models, optimizers, and schedulers to their initial states
            for i in range(self.model_num):
                self.models[i].load_state_dict(self.initial_model_states[i])
                self.optimizers[i].load_state_dict(self.initial_optimizer_states[i])
                #self.schedulers[i].load_state_dict(self.initial_scheduler_states[i])

            # Call the train function to train with the current rho
            self.train_fixed_rho(rho, adversarial=False)
        print("Finished training student cohort!")

        print("Selecting the optimal disgreement penalty via cross-validation...")
        self.choose_best_rho()
        print("Done!")

    def test_adaptive(self, test_loader):
        # Make sure the best model is loaded
        self.ckpt_dir = self.ckpt_dir_adaptive   # set the correct checkpoint saving directory
        #best_val_task_losses = self.load_checkpoints(self.best_rho)
        best_val_task_losses = self.load_checkpoints(self.fixed_rho)
        
        if self.task_type == "classification":
            results = self.test_classification(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        else:
            results = self.test_regression(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        
        return results

    def train_adversarial(self):
        """
        Training with adversarial weights
        """

        self.ckpt_dir = self.ckpt_dir_adversarial   # set the correct checkpoint saving directory
        self.clustering_weights, self.cluster_idxs = None, None  # initialize weights

        # Reset models, optimizers, and schedulers to their initial states
        for i in range(self.model_num):
            self.models[i].load_state_dict(self.initial_model_states[i])
            self.optimizers[i].load_state_dict(self.initial_optimizer_states[i])

        self.train_fixed_rho(self.fixed_rho, adversarial=True)

    
    def test_adversarial(self, test_loader):
        """
        Main testing function that performs the following:
        1. Load the cohort with fixed disagreement penalty
        2. Evaluate the method performance (including various user-specified ensembles) on the test data.
        """

        # Make sure the adversarial models are loaded
        self.ckpt_dir = self.ckpt_dir_adversarial   # set the correct checkpoint saving directory
        best_val_task_losses = self.load_checkpoints(self.fixed_rho)
        
        if self.task_type == "classification":
            results = self.test_classification(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        else:
            results = self.test_regression(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        
        results = {f"adversarial_{key}": value for key, value in results.items()}
        return results


    def train_fixed_rho(self, rho, adversarial):
        """
        Train the models for a fixed level of disagreement penalty. 
        """
        
        # Variable to keep track of the best validation loss
        self.best_val_task_losses = [float('inf')] * self.model_num

        if self.divergence_weight_type == "clustering":
            if rho > 0 and self.clustering_weights is None:   
                # use the validation performance of the models under rho = 0
                try:   # Assume indepedent cohort is already trained and val losses are saved to self.rho0_losses
                    if adversarial:
                        if self.verbose:
                            print("Computing adversarial weights by clustering method...")

                        self.clustering_weights, optim_k, self.cluster_idxs = self.get_adversarial_weights_by_clustering(self.rho0_losses, optimal_k=self.optimal_k, 
                                                                                    verbose=self.verbose, random_state=self.random_state)
                        if self.verbose:
                            print(f"Computed adversarial weights by clustering method, weights are {self.clustering_weights}")
                    
                    else:
                        if self.verbose:
                            print("Computing divergence weights by clustering method...")

                        self.clustering_weights, optim_k, self.cluster_idxs = get_weights_by_clustering(self.rho0_losses, optimal_k=self.optimal_k, 
                                                                                 verbose=self.verbose, random_state=self.random_state)  
                        if self.verbose:
                            print(f"Computed divergence weights by clustering method, weights are {self.clustering_weights}")

                    self.optimal_k = optim_k   # update the optimal cluster number
                    weights=self.clustering_weights

                except Exception as e:
                    # Handle any other exceptions that may occur
                    print(f"An error occurred while computing the clustering weights: {e}, \
                            using unweighted disagreement loss instead.")
                    weights = torch.ones(self.model_num)

            elif rho > 0 and self.clustering_weights is not None:   
                weights = self.clustering_weights
            else:
                weights = torch.ones(self.model_num)

        else:
            weights = torch.ones(self.model_num)

        for epoch in range(self.epochs):

            # for scheduler in self.schedulers:
            #     scheduler.step(epoch)
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )
            
            
            if epoch < self.burn_in_epochs:
                # during burn in epochs, train models independently to avoid negative mutual learning
                train_losses, train_task_losses = self.train_one_epoch(epoch, 0, weights)
                val_losses, val_task_losses = self.validate(epoch, 0)
            else: 
                train_losses, train_task_losses = self.train_one_epoch(epoch, rho, weights)
                val_losses, val_task_losses = self.validate(epoch, rho, weights)
                
            for i in range(self.model_num):
                # Check if the current model has the best validation task loss
                is_best = val_task_losses[i].avg < self.best_val_task_losses[i]
                
                # Print train and validation losses
                msg1 = "model_{:d}: train loss: {:.3f}, train task loss: {:.3f} "
                msg2 = "- val loss: {:.3f}, val task loss: {:.3f}"
                if is_best:
                    msg2 += " [*] Best so far"
                msg = msg1 + msg2
                
                if self.verbose:
                    print(
                        msg.format(i+1, train_losses[i].avg, train_task_losses[i].avg, 
                                        val_losses[i].avg, val_task_losses[i].avg)
                    )

                # If this is the best model so far, update best loss and save checkpoint
                if is_best:
                    self.best_val_task_losses[i] = val_task_losses[i].avg
                    self.save_checkpoint(rho, i,
                        {
                            "epoch": epoch + 1,
                            "disagreement_penalty": rho,
                            "model_state": self.models[i].state_dict(),
                            "optim_state": self.optimizers[i].state_dict(),
                            "best_val_task_loss": self.best_val_task_losses[i],
                        }
                    )


    def get_adversarial_weights_by_clustering(self, losses, max_k=10, optimal_k=None,
                                                    method='silhouette',
                                                    verbose=False, random_state=0):
        """
        Determines adversarial weights based on clustering of task losses.
        
        Parameters:
        - losses: arraylike representing the task loss for each model.
        - max_k: Maximum number of clusters to test for the elbow method.
        - optimal_k: optimal number of clusters. If None, automatically choose by method
        - method: optimal cluster selection method, choose between 'silhouette' and 'elbow'
        - random_state: Random state for reproducibility in KMeans.

        Returns:
        - weights: numpy array of shape (n_models,) with normalized weights for each model.
        """
        # Step 1: Determine the optimal number of clusters
        losses = np.array(losses)
        if optimal_k is None: 
            max_k = np.min([max_k, len(losses)])
            k_list = np.arange(1,max_k + 1)

            # Automatically determine the elbow point where inertia starts to decrease slowly        
            if method == 'silhouette':
                optimal_k = compute_optimal_k_with_silhouette(losses.reshape(-1,1), k_list, random_state=random_state)
            elif method == 'elbow':
                optimal_k = compute_elbow_point(losses.reshape(-1,1), k_list, len(losses),random_state=random_state)

        # Step 2: Perform K-means clustering with the optimal number of clusters
        optimal_k = int(optimal_k)
        kmeans = KMeans(n_clusters=optimal_k, verbose=verbose, random_state=random_state)
        clusters = kmeans.fit_predict(losses.reshape(-1, 1))

        # Step 3: Identify the cluster with the largest average task loss
        cluster_losses = [np.mean(losses[clusters == i]) for i in range(optimal_k)]
        worst_cluster = np.argmax(cluster_losses)

        # Step 4: Assign weights based on cluster membership
        weights = np.zeros_like(losses, dtype=float)
        weights[clusters == worst_cluster] = 1.0

        weights /= np.sum(weights)
        worst_cluster_idxs = np.where(weights != 0)[0]
        return weights, optimal_k, worst_cluster_idxs                