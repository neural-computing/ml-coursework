%% 
% Import Dataset

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
% Build box plot of dataset - standard EDA activities to gain an understanding 
% of what the dataset looks like

boxplot(table2array(predictors),'Labels',predictorNames);
h = gcf;
h.Position(3) = h.Position(3)*2.5;
title('Predictor Box Plots');
%% 
% Next we generate a cross-correlation plot for all features and the response 
% variable. A correlation matrix has also been generated. These can be used to 
% infer which of our features may be good for infering our response variable. 
% Variables that strongly correlate with the response variable are likely good 
% indicatiors whist variables that strongly correlate with other variables are 
% likely to contain shared information. In this case a decision could be made 
% to remove one of these features from the model.

%Create a pairplot to look at cross-correlation
[~,ax] = plotmatrix(table2array(inputTable));
plotSize =width(inputTable);
for k=1:plotSize
ax(k,1).YLabel.String = inputTable.Properties.VariableNames{1,k}; 
ax(k,1).YLabel.Rotation = 25;
ax(k,1).YLabel.HorizontalAlignment = 'right';
ax(plotSize,k).XLabel.String = inputTable.Properties.VariableNames{1,k};
ax(plotSize,k).XLabel.Rotation = 25;
ax(plotSize,k).XLabel.HorizontalAlignment = 'right';

end

%we can also compute the correlation between all features including the response variable 
correlationMatrix = corr(table2array(inputTable));
display(correlationMatrix)
%% 
% Lets also apply T-SNE to allow us to visualise a 2D map of the dataset
% 
% T-SNE creates a non-linear mapping between the high-dimentional dataset and 
% the 2D visual where data points that are close in the high-dimentional space 
% are close within the 2D map. The axis of the map do not have a specific meaning 
% or orientation plus T-SNE is non-convex meaning that different runs converge 
% to different solutions. Therefore we perform T-SNE 3 times and inspect for consistencies. 
% The intuition is that similaries between the plots may indicate structure within 
% the dataset. We also map the response variable as the hue of a data point. This 
% is useful for understanding that within our high-dimentional dataset, there 
% are non-linear structures and moreover data points with similar values for the 
% response variable tend to be neighbours within the high-dimentional space. 

%Apply Stochastic Neighbour Embedding for Visualization of the Raw data
Y = tsne(table2array(predictors),'Distance', "cosine",'Standardize',true,"Perplexity",30);
bins =  floor(max(response)) - ceil(min(response));
scatter(Y(:,1),Y(:,2),15,discretize(response,bins),"filled");
colorbar;
title("T-SNE Plot for visualizing the dataset - Run 1");
%%
%Apply Stochastic Neighbour Embedding for Dimentionality Reduction
Y = tsne(table2array(predictors),'Distance', "cosine",'Standardize',true,"Perplexity",30);
scatter(Y(:,1),Y(:,2),15,discretize(response,bins),"filled");
colorbar;
title("T-SNE Plot for visualizing the dataset - Run 2");
%%
%Apply Stochastic Neighbour Embedding for Dimentionality Reduction
Y = tsne(table2array(predictors),'Distance', "cosine",'Standardize',true,"Perplexity",30);
scatter(Y(:,1),Y(:,2),15,discretize(response,bins),"filled");
colorbar;
title("T-SNE Plot for visualizing the dataset - Run 3");
%% 
% To predict the response variable, we are going to look at 2 models: EnsembleTree 
% and Gaussian Processes. We can use the built in Matlab functions for these and 
% we also want to leverage cross-validation to gain a better understanding of 
% how well our model generalizes. Finally we also want to leverage the "bayesopt" 
% bayesian optimization methods for model hyperparameter tuning and model selection. 
% These can be accessed through the matlab functions but we may need to come back 
% and configure limits for Bayesian Hyperparameter Optimization. That is what 
% this section break is for.

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
% Blog post of what a gaussian process model is: https://towardsdatascience.com/an-intuitive-guide-to-gaussian-processes-ec2f0b45c71d
regressionGP = fitrgp(predictors, response, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));
%% 
% Although we performed cross-validation in the model selection steps above, 
% we want to perform cross-validation to calculate the validation error for the 
% purpose of model comparison. We therefore take the best EnsembleTree and GaussianProcess 
% models and predict the kFoldLoss. This gives us a consistent manor to compare 
% our two tuned models.

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
barh(imp);
yticklabels(predictorNames)
xlabel("Feature Importance")
title("Feature Importance based on EnsembleTree")
%% 
% We can see that 5 for the features are more important than the others. Lets 
% try training again with these 5 most important features, perform xval and evaluate 
% results. 

%Select the top important features and train both models again
topImportantFeatures = 5;
[kImp,I] = maxk(imp,topImportantFeatures);
topKpredictors = inputTable(:, I);
topKpredictorNames = predictorNames(I)
%%
%Optimize Model Hyperparameters using bayesian inference (bayesopt)
%EnsembleTree
regressionEnsembleTopK = fitrensemble(topKpredictors, response, ...
    'Learners', template, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));
%%
%Gaussian Process Regressor
regressionGPTopK = fitrgp(topKpredictors, response, ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName','expected-improvement-plus', 'Kfold', 5));
%%
%Cross Validation Settings
crossValidationMethod = 'KFold';
crossValidationNumFolds = 5;
crossValidationLossFun = 'mse';

% Perform cross-validation
partitionedModel_RFTopK = crossval(regressionEnsembleTopK, crossValidationMethod, crossValidationNumFolds);
partitionedModel_GPTopK = crossval(regressionGPTopK, crossValidationMethod, crossValidationNumFolds);

% Compute validation predictions
validationPredictions_RF = kfoldPredict(partitionedModel_RFTopK);
validationPredictions_GP = kfoldPredict(partitionedModel_GPTopK);

% Compute validation RMSE
validationRMSE_RFTopK = sqrt(kfoldLoss(partitionedModel_RFTopK, 'LossFun', crossValidationLossFun))
validationRMSE_GPTopK = sqrt(kfoldLoss(partitionedModel_GPTopK, 'LossFun', crossValidationLossFun))
%% 
% We can see that there is a drop is RMSE for the EnsembleTree Regressor however 
% it is minimal. This tells us that the 5 most important features are capturing 
% majority of the information within the dataset.

validationRMSE_RF, validationRMSE_RFTopK
validationRMSE_GP, validationRMSE_GPTopK
%% 
%