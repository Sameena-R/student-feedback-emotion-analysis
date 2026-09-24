clc;
clear;
close all;

%% 1. Load dataset
data = readtable("student_feedback_emotion_dataset.xlsx");

%% 2. Get Feedback and Emotion
feedback = string(data.Feedback);
emotion = categorical(data.Emotion);

%% 3. NLP Preprocessing
documents = tokenizedDocument(feedback);

%% 4. Create Bag of Words
bag = bagOfWords(documents);

%% 5. Convert text into numerical features
X = bag.Counts;
Y = emotion;

disp("NLP preprocessing completed successfully.");

%% 6. Train-Test Split
cv = cvpartition(Y, 'HoldOut', 0.2);

XTrain = X(training(cv), :);
XTest  = X(test(cv), :);

YTrain = Y(training(cv));
YTest  = Y(test(cv));

%% 7. Create SVM Model
t = templateLinear('Learner', 'svm');

model = fitcecoc(XTrain, YTrain, ...
    'Learners', t, ...
    'Coding', 'onevsall');

disp("Machine Learning model trained successfully.");

%% 8. Test Model
YPred = predict(model, XTest);

%% 9. Calculate Accuracy
accuracy = mean(YPred == YTest) * 100;

fprintf("Model Accuracy = %.2f%%\n", accuracy);

%% 10. Confusion Matrix
figure;
confusionchart(YTest, YPred);
title("Student Feedback Emotion Classification");

%% 11. New Student Feedback
newFeedback = input("Enter new student feedback: ", "s");

%% 12. Convert new feedback into numerical features
newDocument = tokenizedDocument(string(newFeedback));
newX = encode(bag, newDocument);

%% 13. Predict Emotion
predictedEmotion = predict(model, newX);

fprintf("Predicted Emotion = %s\n", string(predictedEmotion));