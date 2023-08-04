from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.calibration import CalibratedClassifierCV
import shap

human_texts = ['Have fun in life']  
ai_texts = ["Enjoy life's adventures."]  # from chatgpt
labels = [0] * len(human_texts) + [1] * len(ai_texts)  
all_texts = human_texts + ai_texts
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(all_texts)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

parameters = {'C': [0.1, 1, 10]}
scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
classifier = GridSearchCV(LogisticRegression(), parameters, cv=5, scoring=scoring, refit='Accuracy')
classifier.fit(X_train_resampled, y_train_resampled)

best_classifier = classifier.best_estimator_

y_pred = best_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', best_classifier)
])

pipeline.fit(X_train_resampled, y_train_resampled)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(pipeline, features, labels, cv=5, scoring='accuracy')
mean_accuracy = cv_scores.mean()
print(f"Mean cross-validated accuracy: {mean_accuracy:.4f}")

y_pred_full_test = pipeline.predict(X_test)
print("Final Evaluation on Full Test Set:")
print(classification_report(y_test, y_pred_full_test))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

joblib.dump(pipeline, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

pipeline = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

k = 1000
selector = SelectKBest(chi2, k=k)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)

best_classifier_selected = LogisticRegression()
best_classifier_selected.fit(X_train_selected, y_train_resampled)

y_pred_selected = best_classifier_selected.predict(X_test_selected)
print("Evaluation on Selected Features:")
print(classification_report(y_test, y_pred_selected))

calibrated_classifier = CalibratedClassifierCV(best_classifier, cv=5, method='sigmoid')
calibrated_classifier.fit(X_train_resampled, y_train_resampled)

y_pred_prob = calibrated_classifier.predict_proba(X_test)

mean_probs = np.mean(y_pred_prob, axis=0)

plt.figure(figsize=(8, 6))
sns.histplot(mean_probs, bins=10, kde=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.show()

classifiers = [
    ('logistic_regression', best_classifier),
    ('random_forest', RandomForestClassifier()),
    ('gradient_boosting', GradientBoostingClassifier())
]

voting_classifier = VotingClassifier(estimators=classifiers, voting='hard')
voting_classifier.fit(X_train_resampled, y_train_resampled)

y_pred_ensemble = voting_classifier.predict(X_test)

print("Evaluation of Ensemble Classifier:")
print(classification_report(y_test, y_pred_ensemble))

best_classifier.fit(features, labels)

explainer = shap.Explainer(best_classifier)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=vectorizer.get_feature_names())

joblib.dump(voting_classifier, 'ensemble_classifier.joblib')

voting_classifier_loaded = joblib.load('ensemble_classifier.joblib')

y_pred_loaded = voting_classifier_loaded.predict(X_test)
