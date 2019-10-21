close all;
clear all;
inputTable = readtable("../data/Concrete_Data.csv");

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
predictorNames = {'Cement_component1__kgInAM_3Mixture_', 'BlastFurnaceSlag_component2__kgInAM_3Mixture_', ...
    'FlyAsh_component3__kgInAM_3Mixture_', 'Water_component4__kgInAM_3Mixture_', 'Superplasticizer_component5__kgInAM_3Mixture_', ...
    'CoarseAggregate_component6__kgInAM_3Mixture_', 'FineAggregate_component7__kgInAM_3Mixture_', 'Age_day_'};
predictors = inputTable(:, predictorNames);
response = inputTable.ConcreteCompressiveStrength_MPa_Megapascals_;

%Use params = hypermparameters() to set limits for hyperparameter Optimization

%Optimize Model Hyperparameters using bayesian inference
%EnsembleTree
template = templateTree('Reproducible', true);
regressionEnsemble = fitrensemble(predictors, response, ...
    'Learners', template, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));

%Gaussian Process Regressor
regressionGP = fitrgp(predictors, response, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));

%Cross Validation Settings
crossValidationMethod = 'KFold';
crossValidationNumFolds = 5;
crossValidationLossFun = 'mse';

% Perform cross-validation
partitionedModel_RF = crossval(regressionEnsemble, crossValidationMethod, crossValidationNumFolds);
partitionedModel_GP = crossval(regressionGP, crossValidationMethod, crossValidationNumFolds);

% Compute validation predictions
validationPredictions_RF = kfoldPredict(partitionedModel_RF);
validationPredictions_GP = kfoldPredict(partitionedModel_GP);

% Compute validation RMSE
validationRMSE_RF = sqrt(kfoldLoss(partitionedModel_RF, 'LossFun', crossValidationLossFun))
validationRMSE_GP = sqrt(kfoldLoss(partitionedModel_GP, 'LossFun', crossValidationLossFun))

%save trained models and figures 1 & 2
save("trainedModels/regressionEnsemble","regressionEnsemble","validationRMSE_RF");
save("trainedModels/regressionGP","regressionGP","validationRMSE_GP");
saveas(figure(1),'trainedModels/regressionEnsemble_hyperparameter_tuning.png');
saveas(figure(2),'trainedModels/regressionGP_hyperparameter_tuning.png');




