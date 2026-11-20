== Predictive Modeling

=== 1. Objective and Problem Setup

The goal of this study is to predict whether a motor vehicle crash results in a serious or fatal injury (Class 1) versus no injury or a minor injury (Class 0) using 2020 Crash Report Sampling System (CRSS) data. The serious or fatal class includes injury severity codes 3 and 4, which account for 14.45% of our dataset. To prevent data leakage, we excluded the total number of injured occupants (`NO_INJ_IM`) because post-crash injury counts directly reveal the outcome being predicted. Instead, we constructed 20 input features from six factors observable at the time of the crash: weather, lighting, time of day, urban versus rural setting, alcohol involvement, and vehicle count (log-transformed so rare multi-vehicle pileups do not disproportionately skew typical one- and two-vehicle crashes). Using a fixed random seed of 42, we partitioned the 54,544 crashes into 64% for training (34,908 crashes), 16% for validation (8,727 crashes), and 20% for final testing (10,909 crashes). All feature engineering and hyperparameter tuning were performed exclusively on the validation set so that the test set remained strictly held out until final evaluation.

=== 2. Model Setup, Scaling, and Training

Following our course neural-network implementation in Julia, we formulated the logistic regression model using a single linear layer ($h = 1$) with an overflow-safe sigmoid activation function, training it for 3,000 iterations using `Zygote` for automatic differentiation:

$ hat(y) = sigma(bold(w) bold(x) + b) = cases(
  frac(1, 1 + e^(-(bold(w) bold(x) + b))) & "if " bold(w) bold(x) + b >= 0,
  frac(e^(bold(w) bold(x) + b), 1 + e^(bold(w) bold(x) + b)) & "if " bold(w) bold(x) + b < 0
) $

Because our continuous features (`HOUR_IM` and `log1p(VE_TOTAL)`) were already normalized to the $[0, 1]$ range alongside our binary indicator features, gradient descent converged smoothly without additional z-score standardization. To verify this, we also tested z-score standardization (scaling each feature to a mean of 0 and standard deviation of 1), which produced nearly identical validation scores: a Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of 0.6533—which measures how reliably the model ranks severe crashes higher than minor ones across all probability thresholds—and a Precision-Recall Area Under the Curve (PR-AUC) of 0.2629, which measures the trade-off between identifying true severe crashes and avoiding false positives in an imbalanced dataset. As shown in @fig-lr-convergence, a learning rate of 0.20 converged faster than 0.05 without oscillating, stabilizing at a training binary cross-entropy loss of approximately 0.402.

#figure(
  image("assets/fig1_logistic_convergence-1.png", width: 85%),
  caption: [Logistic Regression Training Convergence. A learning rate of 0.20 learns faster than 0.05 and settles at a steady training error of ~0.402.]
) <fig-lr-convergence>

=== 3. Cross-Validation and Regularization Tests

To ensure our validation results were reliable and not dependent on a single random data split, we performed 5-fold cross-validation across the 34,908 training crashes (@fig-lr-cv-pr, left). Across all five folds, the model yielded an average ROC-AUC of 0.6276 ± 0.0095 and an average PR-AUC of 0.2453 ± 0.0112. The small standard deviation across folds (under 0.012) indicates that the model performs consistently across different subsets of the data. We also tested L2 and L1 (Lasso) regularization penalties to check for overfitting. Adding an L2 penalty shrank the feature weights without improving validation ROC-AUC (0.6500 at $lambda = 0$ vs. 0.6419 at $lambda = 0.01$), confirming that the model was not overfitting. Adding an L1 penalty at $lambda = 0.001$ eliminated 13 weak features while keeping validation ROC-AUC nearly unchanged at 0.6481, retaining only seven core predictors (Alcohol Involved, Dark Not Lighted, Dark Lighted, Rural Location, Vehicle Count, Rain, and Snow). Because $lambda = 0$ produced the highest validation score, we retained $lambda = 0$ for the final test evaluation.

=== 4. Final Test Results and Handling Rare Severe Crashes

When evaluated on the 10,909 unseen test crashes, the final model yielded a ROC-AUC of 0.6186 and a PR-AUC of 0.2241, which is 1.64 times higher than the 0.1363 baseline from random guessing (@fig-lr-cv-pr, right). Because all of the model's predicted risk scores fall below 50%, applying the standard 0.500 classification threshold creates a misleading accuracy paradox: the model achieves 86.37% overall accuracy simply by predicting that every crash is minor, thereby failing to identify any serious or fatal crashes (0% recall).

#figure(
  image("assets/fig2_cv_and_pr.png", width: 95%),
  caption: [Cross-Validation ROC Curves (Left) and Final Test Precision-Recall Curve (Right). All five training folds cluster tightly around the average (0.628 ± 0.010), and test PR-AUC (0.224) beats the 0.136 majority baseline.]
) <fig-lr-cv-pr>

#figure(
  image("assets/fig3_logistic_thresholds-1.png", width: 85%),
  caption: [Score Trade-Offs Across Cutoffs. Lowering the cutoff from 0.500 to the validation-best 0.125 trades overall accuracy for higher severe-crash recall and the best F1 score.]
) <fig-lr-thresholds>

To address this class imbalance, we tested two practical strategies on the unseen test set (@tbl-lr-results): lowering the probability threshold required to flag a severe crash (cutoff tuning) versus penalizing the model more heavily for missing severe crashes during training (class-weighted training). First, lowering the decision cutoff from 0.500 to $t = 0.125$—the threshold that maximized the F1 score (the harmonic mean of precision and recall) on the validation set—allows the standard model to flag high-risk crashes. On the test set, this $t = 0.125$ cutoff achieves 57.71% overall accuracy, identifies 64.36% of all serious or fatal crashes (957 out of 1,487), and yields our highest F1 score of 0.2932 (with 18.99% precision). Second, we retrained the model with class weights that penalized missed severe crashes 5.9 times more heavily than minor ones, enabling the use of the default $t = 0.500$ cutoff; this approach results in 68.45% overall accuracy and identifies 43.58% of severe crashes (F1 = 0.2735). For safety screening applications, we recommend standard training with cutoff tuning ($t = 0.125$) because it identifies over 20 percentage points more life-threatening crashes while preserving calibrated risk probabilities that reflect the true baseline rate in the data.

#figure(

  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    [*Model / Strategy*], [*Cutoff ($t$)*], [*Accuracy*], [*Precision*], [*Recall*], [*F1 Score*], [*Confusion Matrix (TP / FP / TN / FN)*],
    [Majority Baseline], [—], [86.37%], [0.00%], [0.00%], [0.0000], [0 / 0 / 9,422 / 1,487],
    [Standard LR (Default)], [0.500], [86.37%], [0.00%], [0.00%], [0.0000], [0 / 0 / 9,422 / 1,487],
    [Standard LR (Conservative)], [0.200], [80.36%], [25.56%], [23.07%], [0.2425], [343 / 999 / 8,423 / 1,144],
    [Standard LR (Balanced)], [0.150], [70.07%], [20.21%], [40.55%], [0.2697], [603 / 2,381 / 7,041 / 884],
    [*Standard LR (Val-Best)*], [*0.125*], [*57.71%*], [*18.99%*], [*64.36%*], [*0.2932*], [*957 / 4,083 / 5,339 / 530*],
    [Class-Weighted LR (5.9× Weight)], [0.500], [68.45%], [19.93%], [43.58%], [0.2735], [648 / 2,603 / 6,819 / 839],
  ),
  caption: [Final Test Set Scores Across Cutoffs and Imbalance Methods ($N = 10,909$).]
) <tbl-lr-results>

=== 5. Which Factors Matter Most

The model's learned weights show how strongly each factor increases or decreases predicted severe injury risk relative to a baseline clear, daytime urban crash (@fig-lr-weights). Alcohol involvement (+1.06) is the strongest positive predictor of severe injury, followed by dark unlit roads (+0.61) and dark lit roads (+0.38). Conversely, vehicle count (−1.10) has the largest negative weight because severe injuries occur more frequently in high-speed single- or two-vehicle collisions than in lower-speed multi-vehicle pileups. To verify which feature groups contribute most to predictive performance, we conducted an ablation study by removing one feature group at a time (@fig-lr-ablation). Removing Vehicle Count (−0.0262), Alcohol (−0.0226), or Lighting (−0.0187) caused the largest decreases in validation ROC-AUC, whereas removing Time of Day slightly increased the score (+0.0009). Finally, adding three pairwise interaction terms (Alcohol × Dark Unlit, Alcohol × Rural, and Dark Unlit × Rural) modestly increased validation PR-AUC (from 0.2623 to 0.2644) while leaving ROC-AUC unchanged (0.6493), indicating that simple two-way feature combinations provide minimal additional predictive gain.

#figure(
  image("assets/fig4_logistic_weights_labeled.png", width: 90%),
  caption: [Learned Logistic Regression Weights. Red bars raise severe injury risk (led by alcohol involvement and dark roads); blue bars lower risk (led by multi-car crashes and snow/rain).]
) <fig-lr-weights>

#figure(
  image("assets/fig6_feature_ablation.png", width: 85%),
  caption: [Feature Ablation Impact on Validation ROC-AUC. Removing vehicle count, alcohol, or lighting causes the biggest drops in validation score, while removing time of day gives a tiny gain (+0.0009).]
) <fig-lr-ablation>

=== 6. Strengths, Limits, and Future Improvements

The primary strengths of this logistic regression model are its interpretability, consistent performance across 5-fold cross-validation (ROC-AUC = 0.6276 ± 0.0095), and ability to reliably rank high-risk crashes above low-risk ones. Its main limitations are that a linear model cannot automatically capture complex multi-way feature interactions without manual feature engineering, and training without survey weights calibrates predicted risk probabilities to our sample's severe crash rate (14.45%) rather than the true national rate (3.64%). Future improvements include removing low-value features such as `HOUR_IM`, applying L1 (Lasso) regularization ($lambda = 0.001$) to produce a more parsimonious seven-feature model, incorporating Crash Report Sampling System (CRSS) survey weights during training to align predicted probabilities with national crash rates, and adding roadway speed-limit data.

