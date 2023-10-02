from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from plotnine import * 
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-d", "--dataset", required=True,
   help="input dataset: Comprising of Hot-encoded variations... subtype, georegion and status")
ap.add_argument("-c", "--comments", required=True,
   help="Stanford HIVdb drug resistance comments")
args = vars(ap.parse_args())

variants_data_path = str(args['dataset'])
stanford_muts_path = str(args['comments'])

variants_data = pd.read_csv(variants_data_path)
#variants_data = pd.read_csv("losAlomosData4ML.csv")
#variants_data.head()
stanford_muts = pd.read_csv(stanford_muts_path)
#stanford_muts = pd.read_csv('Stanford-resistance-comments.csv')
stanford_mut_list = set(stanford_muts['Mutation'])
variants_data.shape

# Select only columns that are not in the exclude_cols list
selected_cols = variants_data.columns.difference(stanford_mut_list)

# Create a new DataFrame with only the selected columns
# variants_data = variants_data[selected_cols]
variants_data.shape

target = 'Status'
drop_vars = ['Status','Year', 'Subtype', 'Georegion']
#drop_vars = ['Status']#,'Sample']
test_size = 0.2
pos_label = 'yes'
perc_top_features = 0.01 #we analyse the top 1% mutations
gene = 'IN'
algor_colors = ["#6082B6","#2AAA8A","#E3963E"]


def get_potential_drms(importance_df, stanford_mut_list):
    imp_muts = importance_df[abs(importance_df.Importance)>0]
    mut_list = set(imp_muts['Feature_names'])
    known_drms = mut_list.intersection(stanford_mut_list)
    potential_drms = mut_list.difference(known_drms)
    return known_drms, potential_drms

def compute_sens_spec(cm):
    tn = cm.confusion_matrix[0,0]
    fp = cm.confusion_matrix[0,1]
    fn = cm.confusion_matrix[1,0]
    tp = cm.confusion_matrix[1,1]
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    return sens, spec

y = variants_data[[target]].values.ravel()
X = variants_data.drop(drop_vars, axis = 'columns')

selector= SelectKBest(score_func=chi2, k=X.shape[1])
fit = selector.fit(X, y)
#X_new=selector.fit_transform(X, y)

selector_dict = dict(mutations=X.columns ,pvalues=fit.pvalues_, scores=fit.scores_)
selector_df = pd.DataFrame(selector_dict)
selector_df_drop = selector_df[selector_df['pvalues']>0.05]
selector_df_drop.shape


drop_mutations = selector_df_drop['mutations']
len(drop_mutations)
X = X.drop(drop_mutations, axis = 'columns')
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
n_top_features = round(perc_top_features*X.shape[1])

# Create a Logistic Regression classifier
logistic_regression = LogisticRegression(solver='liblinear')

# Train the classifier on the training data
logistic_regression.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_regression.predict(X_test)
y_score = logistic_regression.decision_function(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='yes')
recall = recall_score(y_test, y_pred, pos_label='yes')
f1 = f1_score(y_test, y_pred, pos_label='yes')
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

cm = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred))
cm.plot()
sens, spec = compute_sens_spec(cm)
fpr_lr, tpr_lr, thr_lr = roc_curve(y_test, y_score, pos_label=pos_label)
auc_lr = auc(fpr_lr, tpr_lr)

print("Sensitivity: " + str(round(sens,2)))
print("Specificity: " + str(round(spec,2)))
print("AUC: "+str(round(auc_lr,2)))


model_imp = logistic_regression.coef_[0] #this is for SVM linear
feature_names = list(X.columns.values)
importance = list(abs(model_imp))
importance_dict = {'Feature_names':feature_names, 'Importance':importance, 'Algorithm':'LR', 'Gene':gene}
imp_df = pd.DataFrame(importance_dict)
imp_df_lr_ = imp_df.sort_values('Importance', ascending=False)
imp_df_lr = imp_df_lr_.head(n_top_features)
known, potential = get_potential_drms(imp_df, stanford_mut_list)
print("No. of known DRMs: "+str(len(known)))
print("No. of other Mutations: "+str(len(potential)))
print(known)

tmp=imp_df_lr[imp_df_lr['Feature_names'].isin(list(potential))]
tmp=tmp.sort_values('Importance', ascending=False)
toplr=tmp.head(n_top_features)
print(toplr)

# create a random forest classifier

clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_test_pred = clf.predict_proba(X_test)
fpr_rf, tpr_rf, thr_rf = roc_curve(y_test, y_test_pred[:,1], pos_label=pos_label)
auc_rf = auc(fpr_rf, tpr_rf)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred))
sens, spec = compute_sens_spec(cm)
print("Sensitivity: " + str(round(sens,2)))
print("Specificity: " + str(round(spec,2)))
print("AUC: "+str(round(auc_rf,2)))
cm.plot()

model_imp = clf.feature_importances_ #this is for RF and GBM
feature_names = list(X.columns.values)
importance = list(abs(model_imp))
importance_dict = {'Feature_names':feature_names, 'Importance':importance, 'Algorithm':'RF', 'Gene':gene}
imp_df = pd.DataFrame(importance_dict)
imp_df_rf_ = imp_df.sort_values('Importance', ascending=False)
imp_df_rf = imp_df_rf_.head(n_top_features)
known, potential = get_potential_drms(imp_df, stanford_mut_list)
print("No. of known DRMs: "+str(len(known)))
print("No. of other Mutations: "+str(len(potential)))
print(known)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='yes')
recall = recall_score(y_test, y_pred, pos_label='yes')
f1 = f1_score(y_test, y_pred, pos_label='yes')
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

tmp=imp_df_rf_[imp_df_rf_['Feature_names'].isin(list(potential))]
tmp=tmp.sort_values('Importance', ascending=False)
toprf = tmp.head(n_top_features)

# create a GBM classfier

gbm = GradientBoostingClassifier()
gbm.fit(X_train,y_train)
y_pred=gbm.predict(X_test)
y_test_pred = gbm.predict_proba(X_test)
fpr_gbm, tpr_gbm, thr_gbm = roc_curve(y_test, y_test_pred[:,1], pos_label=pos_label)
auc_gbm = auc(fpr_gbm, tpr_gbm)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred))
sens, spec = compute_sens_spec(cm)
print("Sensitivity: " + str(round(sens,2)))
print("Specificity: " + str(round(spec,2)))
print("AUC: "+str(round(auc_gbm,2)))
cm.plot()

model_imp = gbm.feature_importances_ #this is for RF and GBM
feature_names = list(X.columns.values)
importance = list(abs(model_imp))
importance_dict = {'Feature_names':feature_names, 'Importance':importance, 'Algorithm':'GBM', 'Gene':gene}
imp_df = pd.DataFrame(importance_dict)
imp_df_gbm = imp_df.sort_values('Importance', ascending=False)
imp_df_gbm = imp_df_gbm.head(n_top_features)
known, potential = get_potential_drms(imp_df, stanford_mut_list)
print("No. of known DRMs: "+str(len(known)))
print("No. of other Mutations: "+str(len(potential)))
print(known)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='yes')
recall = recall_score(y_test, y_pred, pos_label='yes')
f1 = f1_score(y_test, y_pred, pos_label='yes')
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1:.2f}")
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

tmp=imp_df_gbm[imp_df_gbm['Feature_names'].isin(list(potential))]
tmp=tmp.sort_values('Importance', ascending=False)
topgbm = tmp.head(n_top_features)


rf_gb = set(topgbm.Feature_names).intersection(toprf.Feature_names)
rf_gb_lr = rf_gb.intersection(toplr.Feature_names)

# plot feature importance 

list_imp = [toprf.head(10), toplr.head(10), topgbm.head(10)]
list_imp
imp_df = pd.concat(list_imp)    
imp_df.head()

plot = (ggplot(imp_df)
    + aes(x='Feature_names', y='Importance', fill='Algorithm')
    + geom_bar(stat="identity")
    + facet_wrap('~Algorithm', scales='free') #+ facet_wrap('~Algorithm', scales='free')
    + theme(subplots_adjust={'wspace': 0.15})
    + theme(panel_background=element_rect(fill='white'),
            panel_border=element_rect(color='black',size=1),
            axis_text_x=element_text(color='black', size=15, angle=0),
            axis_text_y=element_text(color='black', size=15),
            axis_title=element_text(color='black', size=20,weight='bold'),
            legend_text=element_text(color='black', size=15),
            legend_title=element_text(color='black', size=20,weight='bold'),
            strip_text=element_text(color='black',size=20,weight='bold'),
            strip_background=element_rect(color='black', fill="white"),
            
      )
    +scale_fill_manual(values = algor_colors)
    +coord_flip()
        +theme(legend_position="none")
    + ylab('Feature importance')
    + xlab('Amino acid mutations'))
plot
plot.save(gene+"-importance-plot.png", width=10, height=8, dpi=300)

# plot the roc curves
roc_dict_lr = {'FPR':fpr_lr, 'TPR':tpr_lr, 'Algorithm':'LR', 'Gene':gene}
roc_dict_lr = pd.DataFrame(roc_dict_lr)
roc_dict_lr

roc_dict_rf = {'FPR':fpr_rf, 'TPR':tpr_rf, 'Algorithm':'RF', 'Gene':gene}
roc_df_rf = pd.DataFrame(roc_dict_rf)
roc_df_rf


roc_dict_gb = {'FPR':fpr_gbm, 'TPR':tpr_gbm, 'Algorithm':'GBM', 'Gene':gene}
roc_df_gb = pd.DataFrame(roc_dict_gb)
roc_df_gb


list_roc = [roc_df_rf, roc_dict_lr, roc_df_gb]
list_roc
roc_df = pd.concat(list_roc)    
roc_df.head()

plot = (ggplot(roc_df)
    + aes(x='FPR', y='TPR', color='Algorithm')
    + geom_line(size=1)
    + facet_wrap('~Algorithm', scales='free') #+ facet_wrap('~Algorithm', scales='free')
    + theme(subplots_adjust={'wspace': 0.15})
    + theme(panel_background=element_rect(fill='white'),
            panel_border=element_rect(color='black',size=1.5),
            axis_text_x=element_text(color='black', size=12, angle=0),
            axis_text_y=element_text(color='black', size=12),
            axis_title=element_text(color='black', size=15,weight='bold'),
            legend_text=element_text(color='black', size=12),
            legend_title=element_text(color='black', size=15,weight='bold'),
            strip_text=element_text(color='black',size=15,weight='bold'),
            strip_background=element_rect(color='black', fill="white"),
            
      )
    +scale_color_manual(values = algor_colors)
        +theme(legend_position="none")
    + ylab('True Positive Rate')
    + xlab('False Positive Rate'))
plot
plot.save(gene+"-ROC-curve.png", width=10, height=8, dpi=300)

# Geographical distibution of sequences

df1 = variants_data[['Georegion','Status']]
# Create a cross-tabulation table with three variables
georegion_map = {'AFR SSA':'Africa', 'Africa':'Africa', 'Asia':'Asia', 'Europe':'Europe',
                  'North America':'America', 'South America':'America', 'Central America':'America'}

# Map levels to values and use 'Others' as the default value
df1['Georegion'] = df1['Georegion'].map(georegion_map).fillna('Others')

df1['Status'] = df1['Status'].map({'no':'Naive', 'yes':'Treated'}).fillna('Others')

df2 = pd.crosstab([df1['Georegion'], df1['Status']], columns=df1['Status'])

# Reset the index to make the categories columns again
df2.reset_index(inplace=True)

# Convert the table to long form using pd.melt()
df3 = df2.melt(id_vars=['Georegion'], value_vars=['Naive', 'Treated'], var_name='Status', value_name='Count')
df3 = df3[df3['Count']>0]
df3.head()


plotg = ((ggplot(df3)
    + aes(x='Georegion', y='Count', fill='Status')
    + geom_bar(stat="identity")
  #  + facet_wrap('~Algorithm', scales='free') #+ facet_wrap('~Algorithm', scales='free')
    + theme(subplots_adjust={'wspace': 0.15})
    + theme(panel_background=element_rect(fill='white'),
            panel_border=element_rect(color='black',size=1),
            axis_text_x=element_text(color='black', size=15, angle=0),
            axis_text_y=element_text(color='black', size=15),
            axis_title=element_text(color='black', size=20,weight='bold'),
            legend_text=element_text(color='black', size=15),
            legend_title=element_text(color='black', size=20,weight='bold'),
            strip_text=element_text(color='black',size=20,weight='bold'),
            strip_background=element_rect(color='black', fill="white"),
            
      )
    +scale_fill_manual(values = algor_colors)
  #  +coord_flip()
        +theme(legend_position="right")
    + ylab('No. of sequences')
    + xlab('Geographical regions')))


# HIV-1 DIVERSITY

df1 = variants_data[['Subtype','Status']]
# Create a cross-tabulation table with three variables

subtype_map = {'A1':'A1', 'B':'B', 'C':'C', 'D':'D', '01_AE':'CRF01_AE', '02_AG':'CRF02_AG',
              '07_BC':'CRF07_BC'}
# Map levels to values and use 'Others' as the default value
df1['Subtype'] = df1['Subtype'].map(subtype_map).fillna('Others')

df1['Status'] = df1['Status'].map({'no':'Naive', 'yes':'Treated'}).fillna('Others')

df2 = pd.crosstab([df1['Subtype'], df1['Status']], columns=df1['Status'])

# Reset the index to make the categories columns again
df2.reset_index(inplace=True)

# Convert the table to long form using pd.melt()
df3 = df2.melt(id_vars=['Subtype'], value_vars=['Naive', 'Treated'], var_name='Status', value_name='Count')
df3 = df3[df3['Count']>0]
df3.head(10)

plots = ((ggplot(df3)
    + aes(x='Subtype', y='Count', fill='Status')
    + geom_bar(stat="identity")
  #  + facet_wrap('~Algorithm', scales='free') #+ facet_wrap('~Algorithm', scales='free')
    + theme(subplots_adjust={'wspace': 0.15})
    + theme(panel_background=element_rect(fill='white'),
            panel_border=element_rect(color='black',size=1),
            axis_text_x=element_text(color='black', size=15, angle=0),
            axis_text_y=element_text(color='black', size=15),
            axis_title=element_text(color='black', size=20,weight='bold'),
            legend_text=element_text(color='black', size=15),
            legend_title=element_text(color='black', size=20,weight='bold'),
            strip_text=element_text(color='black',size=20,weight='bold'),
            strip_background=element_rect(color='black', fill="white"),
            
      )
    +scale_fill_manual(values = algor_colors)
  #  +coord_flip()
        +theme(legend_position="right")
    + ylab('No. of sequences')
    + xlab('HIV-1 subtypes')))

# Further analysis of top polymorphisms

ccd_pms = ['I72V', 'L74M', 'L74I', 'F100Y', 'L101I', 'T124A', 'K136Q', 'D167E']
ctd_pms = ['V20I', 'T218I', 'A265V', 'R269K', 'S283G', 'L234I']
ntd_pms = ['k14R', 'D25E', 'V31I']

known_pms = [ccd_pms, ctd_pms, ntd_pms]

rf_muts = list(toprf.Feature_names)[1:100]

a = set(rf_muts).intersection(ccd_pms)
b = set(rf_muts).intersection(ctd_pms)
c = set(rf_muts).intersection(ntd_pms)

a.union(b).union(c)
