
Predictive Modeling: Crash Injury Severity (Logistic Regression)


1. Objective and Problem Setup

The goal of this study is to predict whether a car crash causes a serious or fatal injury (Class 1) or no injury / minor injury (Class 0) using 2020 CRSS crash data. The serious or fatal group includes severity codes 3 and 4, which make up 14.45% of our data sample. To make fair predictions without cheating, we left out the total number of injured people (NO_INJ_IM) because post-crash injury counts already give away the answer. Instead, we built 20 input features from six factors known at the time of the crash: weather, lighting, time of day, urban or rural area, alcohol involvement, and the number of vehicles (adjusted using a log scale so rare multi-car pileups do not skew typical one- and two-car crashes). Using a fixed random seed of 42, we split the 54,544 crashes into 64% for training (34,908 crashes), 16% for validation (8,727 crashes), and 20% for final testing (10,909 crashes). All model adjustments were done using only the validation set so the test set stayed completely untouched until the end.

2. Model Setup, Scaling, and Training

Following our class neural-network setup in Julia, we built the logistic regression model using a single linear layer and a safe sigmoid function, training it over 3,000 steps with Zygote:

Because our number features (HOUR_IM and log1p(VE_TOTAL)) were already scaled between 0 and 1 alongside our yes/no features, the model trained smoothly without extra z-score scaling. To be sure, we also tested z-score scaling (setting mean = 0 and standard deviation = 1), which gave nearly the exact same validation score (ROC-AUC = 0.6533, PR-AUC = 0.2629). As shown in Figure 1, a learning rate of 0.20 learned faster than 0.05 without jumping around, settling at a steady training error of about 0.402.

#image("assets/fig1_logistic_convergence-1.png")

Figure 1: Logistic Regression Training Convergence. A learning rate of 0.20 learns faster than 0.05 and settles at a steady training error of ~0.402.

3. Cross-Validation and Regularization Tests

To make sure our validation results were reliable and not due to one lucky data split, we ran 5-fold cross-validation across the 34,908 training crashes (Figure 2, left). Across all five folds, the model earned an average ROC-AUC of 0.6276 ± 0.0095 and an average PR-AUC of 0.2453 ± 0.0112. The tiny spread across folds (under 0.012) proves that the model performs consistently across different slices of the data. We also tested L2 and L1 (Lasso) penalties to check for overfitting. Adding an L2 penalty shrank the weights without improving validation ROC-AUC (0.6500 at \(\lambda = 0\) vs. 0.6419 at \(\lambda = 0.01\)), showing the model was not overfitting. Adding an L1 penalty at \(\lambda = 0.001\) dropped 13 weak features while keeping validation ROC-AUC nearly unchanged at 0.6481, leaving only the seven core predictors (Alcohol Involved, Dark Not Lighted, Dark Lighted, Rural Location, Vehicle Count, Rain, and Snow). Because \(\lambda = 0\) gave the highest validation score, we kept \(\lambda = 0\) for our final test. 

4. Final Test Results and Handling Rare Severe Crashes

When we ran our finished model on the 10,909 unseen test crashes, it earned a ROC-AUC score of 0.6186 and a Precision-Recall AUC score of 0.2241, which is 1.64 times better than random guessing at 0.1363 (Figure 2, right). Because all of the model's predicted risk scores fall below 50%, using a standard 0.500 cutoff creates a misleading accuracy trap: the model gets 86.37% accuracy simply by guessing that every single crash is minor, catching 0% of serious or fatal crashes.

#image("assets/fig2_logistic_roc_pr.png")

Figure 2: Cross-Validation ROC Curves (Left) and Final Test Precision-Recall Curve (Right). All five training folds cluster tightly around the average (0.628 ± 0.010), and test PR-AUC (0.224) beats the 0.136 majority baseline.

#image("assets/fig3_logistic_thresholds-1.png")

Figure 3: Score Trade-Offs Across Cutoffs. Lowering the cutoff from 0.500 to the validation-best 0.125 trades overall accuracy for higher severe-crash recall and the best F1 score.

To fix this imbalance, we compared cutoff tuning against class-weighted training on the test set (Table 1). First, picking a lower cutoff of \(t = 0.125\) on the validation set maximizes the F1 score. On the test set, this \(t = 0.125\) cutoff gives 57.71% overall accuracy, catches 64.36% of all serious or fatal crashes (957 out of 1,487), and earns our highest F1 score of 0.2932 (Precision = 18.99%). Second, training a class-weighted model that gave severe crashes 5.9× more weight allowed us to use the default t = 0.500 cutoff, earning 68.45% accuracy and 43.58% recall (F1 = 0.2735). We recommend standard training with cutoff tuning (t = 0.125) for safety screening because it catches over 20% more life-threatening crashes while keeping risk scores aligned with the data's true rate.

Model / Cutoff	Threshold	Accuracy	Precision	Recall	F1	Confusion Matrix (TP / FP / TN / FN)
Majority Baseline	—	86.37%	0.00%	0.00%	0.0000	0 / 0 / 9,422 / 1,487
LR (Default)	0.500	86.37%	0.00%	0.00%	0.0000	0 / 0 / 9,422 / 1,487
LR (Conservative)	0.200	80.36%	25.56%	23.07%	0.2425	343 / 999 / 8,423 / 1,144
LR (Balanced)	0.150	70.07%	20.21%	40.55%	0.2697	603 / 2,381 / 7,041 / 884
LR (Val-Optimal)	0.125	57.71%	18.99%	64.36%	0.2932	957 / 4,083 / 5,339 / 530

Table 1: Final Test Set Scores Across Cutoffs and Imbalance Methods (N = 10,909).

5. Which Factors Matter Most

The model's learned weights show how much each factor raises or lowers severe injury risk compared to a clear, daytime crash in a city (Figure 4). Alcohol involvement (+1.06) is the biggest driver of severe injuries, followed by dark unlit roads (+0.61) and dark lit roads (+0.38). Conversely, vehicle count (−1.10) has the strongest negative weight because severe injuries mostly occur in high-speed one- or two-car crashes rather than slower multi-car pileups. To confirm which factors actually drive accuracy, we ran a feature-removal test by dropping one factor group at a time (Figure 5). Dropping Vehicle Count (−0.0262), Alcohol (−0.0226), or Lighting (−0.0187) caused the largest drops in validation ROC-AUC, while dropping Time of Day slightly improved the score (+0.0009). Finally, adding three combined interaction features (Alcohol × Dark Unlit, Alcohol × Rural, and Dark Unlit × Rural) slightly raised validation PR-AUC (0.2623 to 0.2644) while leaving ROC-AUC unchanged (0.6493), showing that simple two-way combinations only modestly sharpen precision.

#image("assets/fig4_logistic_weights_labeled.png")

Figure 4: Learned Logistic Regression Weights. Red bars raise severe injury risk (led by alcohol involvement and dark roads); blue bars lower risk (led by multi-car crashes and snow/rain).

#image("assets/fig6_feature_ablation.png")

Figure 5: Feature Ablation Impact on Validation ROC-AUC. Removing vehicle count, alcohol, or lighting causes the biggest drops in validation score, while removing time of day gives a tiny gain (+0.0009).

6. Strengths, Limits, and Future Improvements

The main strengths of this logistic regression model are that it is easy to understand, gives steady results across 5-fold cross-validation (0.6276 ± 0.0095), and reliably ranks high-risk crashes above low-risk ones. Its main limits are that a straight-line model cannot spot complex three-way combinations on its own, and training without survey weights matches our risk scores to our sample rate (14.45%) rather than the national severe crash rate (3.64%). To improve this logistic regression model in the future, we could drop noisy features like HOUR_IM, use L1 Lasso (λ = 0.001) to build a simpler 7-feature model, add national CRSS survey weights during training so predicted risk scores match national rates, and include road speed-limit data.

(Note for Data Exploration Update: During modeling, the strong negative weight on vehicle count (−1.10) and large ablation drop (−0.0262) led us to re-check our earlier data exploration, confirming that serious and fatal injuries happen mostly in high-speed one- and two-vehicle crashes rather than multi-car pileups in slower traffic.)