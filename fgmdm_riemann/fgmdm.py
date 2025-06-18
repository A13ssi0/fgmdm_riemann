# #!/usr/bin/env python

import numpy as np
from pyriemann.classification import FgMDM as fgmdm
from sklearn import svm
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, log_loss
from pyriemann.utils.tangentspace import tangent_space
import matplotlib.pyplot as plt

# model.centroids_  # shape (n_classes, n_features)

class FgMDM:

    def __init__(self,njobs=1):
        self.njobs = njobs
        self.mdl = []



    # train fgmdm model for each band
    # data = centered cov matrices
    def train(self, data, labels, classes, idx_train, idx_val):#, ref_matrix, n_components_W=None):
        self.classes = np.array(classes)

        self.n_bands = data.shape[0]
        self.n_classes = len(classes)

        self.n_channels = data.shape[2]

        for bId in range(self.n_bands):
            cov_train = data[bId, idx_train]
            lbl_train = labels[idx_train]

            model = fgmdm(n_jobs=self.njobs)
            model.fit(cov_train, lbl_train)

            self.mdl.append(model)

        # self.evaluate_on_validation(data[:, idx_val], labels[idx_val])
        self.evaluate_on_training(data[:, idx_train], labels[idx_train], classes)


    def evaluate_on_validation(self, data, labels, classes):
        print(' - Evaluating models on validation set')
        self.val_classification = self.predict(data)
        self.val_probabilities = self.predict_probabilities(data)
        self.val_distances = self.get_distances_fromCentroids(data)
        self.val_accuracies = self.get_accuracies(self.val_classification, labels, classes)
        #self.val_kappa = self.get_kappa_values(self.val_classification, labels)
        self.val_confMatrix = self.get_confusion_matrix(self.val_classification, labels)
        #self.val_log_loss = self.get_log_loss(self.val_classification, labels)
        self.get_merge_grid()
        #self.evaluate_on_merged_models(data, labels)



    def evaluate_on_training(self, data, labels, classes):
        print(' - Evaluating models on training set')
        self.train_classification = self.predict(data)
        self.train_probabilities = self.predict_probabilities(data)
        self.train_distances = self.get_distances_fromCentroids(data)
        self.train_accuracies = self.get_accuracies(self.train_classification, labels, classes)
        #self.val_kappa = self.get_kappa_values(self.val_classification, labels)
        self.train_confMatrix = self.get_confusion_matrix(self.train_classification, labels)
        #self.val_log_loss = self.get_log_loss(self.val_classification, labels)
        #self.get_merge_grid()
        #self.merge_grid = self.train_accuracies.copy() 
        #self.evaluate_on_merged_models(data, labels)
        


    def predict(self, data):
        pred_proba = self.predict_probabilities(data)
        pred_class = np.zeros(pred_proba.shape[0:2], dtype=int)
        for bId in range(self.n_bands):
            cl = np.argmax(pred_proba[bId], axis=1)
            pred_class[bId] = self.classes[cl]
        return pred_class



    def predict_probabilities(self, data):
        n_samples = data.shape[1]
        pred_proba = np.empty((self.n_bands, n_samples, self.n_classes), dtype=float)
        for bId in range(self.n_bands):
            model = self.mdl[bId]
            pred_proba[bId] = model.predict_proba(data[bId])
        return pred_proba
    
    

    def get_distances_fromCentroids(self, data):
        n_samples = data.shape[1]
        distances = np.empty((self.n_bands, n_samples, self.n_classes),dtype=float)
        for bId in range(self.n_bands):
            model = self.mdl[bId]
            distances[bId] =  model.transform(data[bId])
        return distances
    


    def get_accuracies(self, pred_class, true_class, classes):
        return [
            np.sum((pred_class==true_class)[0][(true_class==classes[0])], axis=0)/np.sum((true_class==classes[0])),
            np.sum((pred_class==true_class)[0][(true_class==classes[1])], axis=0)/np.sum((true_class==classes[1]))
        ]



    def get_confusion_matrix(self, pred_class, true_class):
        confMatrix = np.empty((self.n_bands, self.n_classes, self.n_classes),dtype=int)
        for bId in range(self.n_bands):
            confMatrix[bId] = confusion_matrix(true_class, pred_class[bId])
        return confMatrix



    def get_merge_grid(self):
        #self.merge_grid = np.zeros((self.n_bands,1))
        self.merge_grid = self.val_accuracies.copy()   

    
    # merge the output of two classifiers
    def merge_bands(self, pred_proba, labels=None, idx_subtrain=None, idx_subval=None, method='weighted_avg'):
        if pred_proba.shape[0]>1:
            n_sample = pred_proba.shape[1]
            merged_proba = np.empty((1, n_sample, self.n_classes))
            #print('Merging probabilities:')
            if method=='weighted_avg':
                # Weighted avg (weight = band accuracy)
                pred = pred_proba.transpose(1, 0, 2)  # nSample x nBands x nClass
                t_merged = self.merge_grid.T  @ pred / sum(self.merge_grid) 
                merged_proba[0] = t_merged
                merged_class = np.argmax(t_merged, axis=1)
                merged_class = self.classes[merged_class]
                if labels is not None:
                    print('Weighted average----------')
                    print(confusion_matrix(merged_class, labels))

            #elif method=='avg':
                t_merged = np.mean(pred_proba, axis=0)
                #merged_proba[0] = t_merged
                merged_class = np.argmax(t_merged, axis=1)
                merged_class = self.classes[merged_class]
                if labels is not None:
                    print('Average----------')
                    print(confusion_matrix(merged_class, labels))
            
            #elif method=='stacking':
                if idx_subtrain is not None and idx_subval is not None:
                    
                    # clf = StackingClassifier(estimators=[('band1', self.mdl[0]), ('band2', self.mdl[1])], # estimators will be refitted
                    #                         final_estimator=LogisticRegression())
                    # t_merged = clf.fit(cov_events[idx_val[idx_subtrain]], labels[idx_subtrain]).predict_proba(cov_events[idx_val[idx_subval], labels[idx_subval])
                    

                    # Use a training and a validation subsets obtained as the original validation set
                    clf = LogisticRegression()
                    t_merged = clf.fit(pred_proba[:, idx_subtrain, 0].T, labels[idx_subtrain]).predict_proba(pred_proba[:, idx_subval, 0].T)
                    
                    # #merged_proba[0] = t_merged
                    merged_class = np.argmax(t_merged, axis=1)
                    merged_class = self.classes[merged_class]
                    if labels is not None:
                        print('Stacking (logistic) ----------')
                        print(confusion_matrix(merged_class, labels[idx_subval]))

                    # import matplotlib.pyplot as plt
                    # plt.scatter(pred_proba[0, idx_subval[merged_class==773], 0], pred_proba[1, idx_subval[merged_class==773], 0], c=labels[idx_subval[merged_class==773]]-770, alpha=0.5, marker='v')
                    # plt.scatter(pred_proba[0, idx_subval[merged_class==771], 0], pred_proba[1, idx_subval[merged_class==771], 0], c=labels[idx_subval[merged_class==771]]-770, alpha=0.5, marker='o')
                    
                    # Use a training and a validation subsets obtained as the original validation set
                    clf = svm.SVC()
                    merged_class = clf.fit(pred_proba[:, idx_subtrain, 0].T, labels[idx_subtrain]).predict(pred_proba[:, idx_subval, 0].T)
                    
                    if labels is not None:
                        print('Stacking (SVM) ----------')
                        print(confusion_matrix(merged_class, labels[idx_subval]))

            return merged_proba 
        else:
            return pred_proba
    




    def get_centroids(self, Cref=[]):
        if len(Cref)==0:
            Cref = np.eye(self.n_channels)

        centroids = np.empty((self.n_bands, self.n_classes, self.n_channels, self.n_channels))
        for bId in range(self.n_bands):
            centroids[bId] = np.array(self.mdl[bId]._mdm.covmeans_)
       
        tan_centroids = tangent_space(centroids, Cref=Cref)
        return [centroids, tan_centroids]
    











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
    


