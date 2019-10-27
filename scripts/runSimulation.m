%% 
% Initialise Dataset

close all;
clear all;
inputTable = readtable("../data/Concrete_Data.csv");

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
predictorNames = inputTable.Properties.VariableNames(1,1:8);
predictors = inputTable(:, predictorNames);
response = inputTable.Concrete_Compressive_Strength_Mpa;

%Print out head of table
display(inputTable(1:10,:))
%% 
% Apply T-SNE to allow us to visualise the dataset

%Apply Stochastic Neighbour Embedding for Dimentionality Reduction
Y = tsne(table2array(predictors),'Distance', "cosine",'Standardize',true,"Perplexity",50);
bins = 20;
max(response)
min(response)
scatter(Y(:,1),Y(:,2),15,discretize(response,bins),"filled");
%%
%Create a pairplot to look at cross-correlation

[~,ax] = plotmatrix(table2array(inputTable));
plotSize =width(inputTable);
for k=1:plotSize
ax(k,1).YLabel.String = inputTable.Properties.VariableNames{1,k}; 
ax(k,1).YLabel.Rotation = 45;
ax(k,1).YLabel.HorizontalAlignment = 'right';
ax(plotSize,k).XLabel.String = inputTable.Properties.VariableNames{1,k};
ax(plotSize,k).XLabel.Rotation = 45;
ax(plotSize,k).XLabel.HorizontalAlignment = 'right';

end

%we can also compute the correlation between all features including the response variable 
correlationMatrix = corr(table2array(inputTable));
display(correlationMatrix)
%% 
% Configure limits for Bayesian Hyperparameter Optimization

%Use params = hypermparameters() to set limits for hyperparameter Optimization

%% 
% Train RandomForest Model and perform Model Selection

%Optimize Model Hyperparameters using bayesian inference (bayesopt)
%EnsembleTree
template = templateTree('Reproducible', true,'Surrogate','on');
regressionEnsemble = fitrensemble(predictors, response, ...
    'Learners', template, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));
%% 
% Train Gaussian Processes Model and perform Model Selection

%Gaussian Process Regressor
regressionGP = fitrgp(predictors, response, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));
%% 
% Although we performed cross-validation in the model selection steps above, 
% we want to perform cross-validation to calculate the validation error for the 
% purpose of model comparisons

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
%% 
% Save the Trained Models and Model Selection Plots.

%save trained models and figures 1 & 2
%save("trainedModels/regressionEnsemble","regressionEnsemble","validationRMSE_RF");
%save("trainedModels/regressionGP","regressionGP","validationRMSE_GP");
%saveas(figure(3),'trainedModels/regressionEnsemble_hyperparameter_tuning.png');
%saveas(figure(4),'trainedModels/regressionGP_hyperparameter_tuning.png');
%% 
% With EnsembleTree we can look to understand the feature Importance.

[imp,ma] = predictorImportance(regressionEnsemble)
%%

%% 
%