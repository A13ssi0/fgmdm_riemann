# #!/usr/bin/env python

import numpy as np
from pyriemann.classification import FgMDM as fgmdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, log_loss
from pyriemann.utils.tangentspace import tangent_space, log_map_riemann, exp_map_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.test import is_sym_pos_def
import warnings



class FgMDM:

    def __init__(self,njobs=1):
        self.njobs = njobs
        self.mdl = []


    def train(self, data, labels, classes, idx_train=[], idx_val=[]):#, ref_matrix, n_components_W=None):
        if len(idx_train) == 0:     idx_train = np.ones(data.shape[1], dtype=bool)

        self.n_classes = len(classes)
        self.classes = np.array(classes)

        self.n_bands = data.shape[0]
        self.n_channels = data.shape[2]

        self.trainCF = np.empty((self.n_bands, self.n_classes, self.n_classes), dtype=int)

        for bId in range(self.n_bands):
            cov_train = data[bId, idx_train]
            lbl_train = labels[idx_train]

            model = fgmdm(n_jobs=self.njobs)
            model.fit(cov_train,lbl_train)

            self.mdl.append(model)

            t_cov_train = np.expand_dims(cov_train, axis=0)
            self.trainCF[bId] = self.get_confusion_matrix(self.predict(t_cov_train), lbl_train)
            print_confusion_matrix(self.trainCF[bId], labels=[str(cl) for cl in self.classes])

            print(f' - Model accuracy for band {bId+1}/{self.n_bands}: {self.get_accuracies(self.predict(t_cov_train), lbl_train)[bId]:.3f}')

            

        if len(idx_val)>0:  self.evaluate_on_validation(data[:,idx_val], labels[idx_val])


    def evaluate_on_validation(self, data, labels):
        print(' - Evaluating models on validation set')
        self.val_classification = self.predict(data)
        self.val_probabilities = self.predict_probabilities(data)
        self.val_distances = self.get_distances_fromCentroids(data)
        self.val_accuracies = self.get_accuracies(self.val_classification, labels)
        #self.val_kappa = self.get_kappa_values(self.val_classification, labels)
        self.val_confMatrix = self.get_confusion_matrix(self.val_classification, labels)
        #self.val_log_loss = self.get_log_loss(self.val_classification, labels)
        self.get_merge_grid()
        #self.evaluate_on_merged_models(data, labels)
        


    def predict(self,data):
        pred_proba = self.predict_probabilities(data)
        pred_class = np.zeros(pred_proba.shape[0:2], dtype=int)
        for bId in range(self.n_bands):
            cl = np.argmax(pred_proba[bId],axis=1)
            pred_class[bId] = self.classes[cl]
        return pred_class


    def predict_probabilities(self,data):
        n_samples = data.shape[1]
        pred_proba = np.empty((self.n_bands, n_samples, self.n_classes), dtype=float)
        for bId in range(self.n_bands):
            model = self.mdl[bId]
            pred_proba[bId] = model.predict_proba(data[bId])
        return pred_proba
    
    
    def get_distances_fromCentroids(self,data):
        n_samples = data.shape[1]
        distances = np.empty((self.n_bands, n_samples, self.n_classes),dtype=float)
        for bId in range(self.n_bands):
            model = self.mdl[bId]
            distances[bId] =  model.transform(data[bId])
        return distances
    

    def get_accuracies(self, pred_class, true_class):
        return np.sum(pred_class==true_class,axis=1)/len(true_class)


    def get_confusion_matrix(self, pred_class, true_class):
        confMatrix = np.empty((self.n_bands, self.n_classes, self.n_classes),dtype=int)
        for bId in range(self.n_bands):
            confMatrix[bId] = confusion_matrix(true_class, pred_class[bId])
        return confMatrix


    def get_merge_grid(self):
        #self.merge_grid = np.zeros((self.n_bands,1))
        self.merge_grid = self.val_accuracies.copy()   

    

    def merge_bands(self, pred_proba):
        n_sample = pred_proba.shape[1]
        merged_proba = np.empty((1, n_sample, self.n_classes))
        pred = pred_proba.transpose(1,0,2)  # nSample x nBand x nClass
        t_merged = self.merge_grid.T  @ pred / sum(self.merge_grid)
        merged_proba[0] = t_merged
        return merged_proba 
    

    def get_centroids(self, Cref=[]):
        if len(Cref)==0:
            Cref = np.eye(self.n_channels)

        centroids = np.empty((self.n_bands, self.n_classes, self.n_channels, self.n_channels))
        for bId in range(self.n_bands):
            centroids[bId] = np.array(self.mdl[bId]._mdm.covmeans_)
       
        tan_centroids = tangent_space(centroids, Cref=Cref)
        return [centroids, tan_centroids]
    
    def update(self, update_batch, alpha=0.98, maxiter=25, tol=0.01, fullLogMap=True):
        # COVS HANNO LA PRIMA E SECONDA DIMENSIONE INVERTITE, DOVREBBERE ESSERE N_BANDS X N_SAMPLES X N_CHANNELS X N_CHANNELS
        # SE TI SERVE FAI UN np.transpose(covs, (1,0,2,3)) RICORDA DI CAMBIARE GLI INDICI DOPO PERÒ!!! No no, era per ricordarmi questa cosa degli indici invertiti

        covs, lbls = update_batch.get_batch()
        for bId in range(self.n_bands):
            new_cov = self.mdl[bId]._fgda.transform(covs[:,bId,:,:])
            covs[:,bId,:,:] = new_cov

        for clssId, clss in enumerate(self.classes): 
            idx = (lbls==clss)
            if sum(idx) == 0: continue #FIXME NON SO SE SERVE FARE QUESTO IF TANTO SE L'INDICE RESTA VUOTO NON FA PRATICAMENTE NULLA.. PROVA ANCHE SENZA
            for bId in range(self.n_bands):
                new_centroid = mean_riemann(covs[idx,bId,:,:], maxiter=maxiter)#.reshape(1, self.n_channels, self.n_channels) # AGGIUNGO LA DIMENSIONE DELLE BANDE

                old_centroid = self.mdl[bId]._mdm.covmeans_[clssId]
                distance = distance_riemann(new_centroid, old_centroid)

                # # set numpy print options to show first 3 decimals
                # np.set_printoptions(precision=3, suppress=True)
                # print(distance)
                # print(old_centroid)
                # print(new_centroid)

                
                updated_centroid = alpha*old_centroid + (1-alpha)*new_centroid # inizializzazione
                distance_updated = distance_riemann(updated_centroid, old_centroid)
                iter_counter = 0

                while abs(distance_updated/distance - (1 - alpha)) > tol:
                    if iter_counter > maxiter:
                        warnings.warn(f' - WARNING: maxiter reached for class:{clss} and bId:{bId}.')  
                        break

                    tan_updated_centroid = log_map_riemann(updated_centroid, Cref=updated_centroid, C12=fullLogMap)
                    if distance_updated/distance > (1 - alpha): tan_objective_centroid = log_map_riemann(old_centroid, Cref=updated_centroid, C12=fullLogMap) # da portare verso old centroid
                    else:                                       tan_objective_centroid = log_map_riemann(new_centroid, Cref=updated_centroid, C12=fullLogMap) # da portare verso new centroid
                    directional_matrix = tan_objective_centroid - tan_updated_centroid  # like vector that connects the two points
                    tan_updated_centroid += abs(distance_updated/distance - (1 - alpha))*directional_matrix  # to 'recover' the distance that is missing
                    updated_centroid = exp_map_riemann(tan_updated_centroid, Cref=updated_centroid, Cm12=fullLogMap)  # to go back to the manifold
                    distance_updated = distance_riemann(updated_centroid, old_centroid)
                    iter_counter += 1
                    
                    
                if not is_sym_pos_def(updated_centroid):    # non dovrebbe mai succedere, ma si sa mai
                    warnings.warn(f' - WARNING: updated centroid is not SPD for class:{clss} and bId:{bId}. Keeping the old centroid')  
                    updated_centroid = old_centroid
                
                self.mdl[bId]._mdm.covmeans_[clssId] = updated_centroid
        print(f'Classifiers updated. Distance updated: {distance_updated:.5f} - distance: {distance:.5f} - rapporto: {1-distance_updated/distance:.5f}')
               

# -----------------------------------------------------------------------------------------------------

from rich.console import Console
from rich.table import Table
from rich import box

def print_confusion_matrix(matrix, labels=None):
    console = Console()
    n = len(matrix)
    table = Table(show_header=True, header_style="bold bright_cyan", box=box.SIMPLE_HEAVY)

    # Add column headers
    table.add_column(" ", style="bold bright_cyan")
    for i in range(n):
        label = labels[i] if labels else str(i)
        table.add_column(f"P_{label}", justify="center")

    # Add matrix rows
    max_val = matrix.max()
    for i in range(n):
        label = labels[i] if labels else str(i)
        row = [f"[bright_yellow]T_{label}[/]"]
        for j in range(n):
            val = matrix[i][j]
            ratio = val / max_val if max_val else 0

            # Bright color gradient
            if i == j:
                color = "bold bright_green"
            elif ratio > 0.66:
                color = "bold bright_red"
            elif ratio > 0.33:
                color = "bright_magenta"
            else:
                color = "bright_black"

            row.append(f"[{color}]{val}[/]")
        table.add_row(*row)

    console.print(table)





    # def merge_classifiers(self, pred_proba):  # every classifier is for only one class (1vsNot1). After is 1vsOthers
    #     n_sample = pred_proba.shape[1]
    #     merged_proba = np.empty((self.n_bands, n_sample, self.n_classes))

    #     for cl in range(self.n_classes):
    #         pred = (1-pred_proba)/(self.n_classes-1)
    #         pred[:,:,cl] = pred_proba[:,:,cl]
    #         pred = pred.transpose(1,0,2)  # nSample x nBand x nClass
    #         t_merged = np.diagonal(pred @ self.merge_grid.T / np.sum(self.merge_grid,axis=1), axis1=1, axis2=2)
    #         merged_proba[:,:,cl] = t_merged.T

    #     return merged_proba




    # def evaluate_on_merged_models(self, data, labels):
    #     pred_proba = self.predict_probabilities(data)

    #     p_mergedClass = self.merge_classifiers(pred_proba)
    #     p_mergedBands = self.merge_bands(pred_proba)

    #     cl_mergedClass =  self.classes[np.argmax(p_mergedClass,axis=2)]
    #     self.mergedClass_accuracy = np.sum(cl_mergedClass==labels,axis=1)/len(labels)
    #     self.mergedClass_confMatrix = np.empty((self.n_bands,self.n_classes,self.n_classes))
    #     for bId in range(self.n_bands):
    #         self.mergedClass_confMatrix[bId] = confusion_matrix(labels,cl_mergedClass[bId])

    #     self.mergedBands_accuracy = np.empty((1,self.n_classes))
    #     self.mergedBands_confMatrix = np.empty((self.n_classes,self.n_classes,self.n_classes))
    #     for cl in range(self.n_classes):
    #         cl_mergedBands = p_mergedBands[0,:,cl]>0.5
    #         t_labels = labels==self.classes[cl]
    #         self.mergedBands_accuracy[0,cl] = np.sum(cl_mergedBands==t_labels)/len(labels)
    #         self.mergedBands_confMatrix[cl] = confusion_matrix(t_labels,cl_mergedBands)



#     def predict_mergeClassifier(self,data):
#         pred_proba = self.predict_probabilities_mergeClassifier(data)
#         pred_class = np.zeros((data.shape[0],data.shape[-1]),dtype=int)
#         for bId in range(pred_class.shape[-1]):
#             idx_max = np.argmax(pred_proba[:,:,bId],axis=1)
#             pred_class[:,bId] = [self.classes[x] for x in idx_max]
#         return pred_class
    

#     def get_accuracies_mergeClassifier(self, pred_class, true_class):
#         accuracies = np.empty(pred_class.shape[-1], dtype=float)
#         for bId in range(accuracies.shape[-1]):
#             correct = pred_class[:,bId]==true_class
#             accuracies[bId] = sum(correct)/len(correct)
#         return accuracies
    



#     #  --------------------------------------
#     def get_kappa_values(self, pred_class, true_class):
#         score = np.empty((self.n_classes, pred_class.shape[-1]), dtype=float)
#         for bId in range(self.n_bands):
#             for j,cl in enumerate(self.classes):
#                 t_class = true_class.copy()
#                 t_class[t_class!=cl] = 0
#                 score[j,bId] = cohen_kappa_score(t_class, pred_class[:,j,bId])
#         if len(score[score<0])>0:
#             print('   - ' + str(len(score[score<0])) + ' negative Kappa values found on validation. Setting to zeros')
#             score[score<0] = 0
#         return score
    
#     def get_log_loss(self, pred_class, true_class):
#         loss = np.empty((self.n_classes, pred_class.shape[-1]), dtype=float)
#         for bId in range(self.n_bands):
#             for j,cl in enumerate(self.classes):
#                 t_class = true_class.copy()
#                 t_class[t_class!=cl] = 0
#                 loss[j,bId] = log_loss(t_class, pred_class[:,j,bId])
#         return loss
    


