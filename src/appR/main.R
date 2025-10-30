
main <- function() {
  cat("=== Titanic Logistic Regression (accuracy only) ===\n")
  
  # ---------- Load ----------
  cat("\n[1] Loading datasets from ./data ...\n")
  train <- read.csv("data/train.csv", stringsAsFactors = FALSE)
  test  <- read.csv("data/test.csv",  stringsAsFactors = FALSE)
  gender_submission <- read.csv("data/gender_submission.csv", stringsAsFactors = FALSE)
  cat(sprintf("train.shape: (%d, %d)\n", nrow(train), ncol(train)))
  cat(sprintf("test.shape:  (%d, %d)\n", nrow(test),  ncol(test)))
  cat(sprintf("gender_submission.shape: (%d, %d)\n", nrow(gender_submission), ncol(gender_submission)))
  
  # ---------- Define X_train, y_train, X_test, y_test ----------
  cat("\n[2] Preparing data ...\n")
  y_train <- train$Survived
  X_train <- subset(train, select = setdiff(colnames(train), c("Survived", "PassengerId")))
  pid_test <- test$PassengerId
  X_test  <- subset(test, select = setdiff(colnames(test), "PassengerId"))
  idx <- match(pid_test, gender_submission$PassengerId)
  y_test <- gender_submission$Survived[idx]
  
  # ---------- Encoding ----------
  cat("\n[3] Encoding categorical variables ...\n")
  
  # Sex
  if ("Sex" %in% colnames(X_train)) {
    cat(" - Encoding 'Sex'\n")
    map_sex <- function(x) ifelse(x == "male", 0,
                                  ifelse(x == "female", 1, NA_real_))
    X_train$Sex <- map_sex(X_train$Sex)
    X_test$Sex  <- map_sex(X_test$Sex)
  }
  
  # Embarked
  if ("Embarked" %in% colnames(X_train)) {
    cat(" - One-hot encoding 'Embarked' (drop_first=True)\n")

    X_train$Embarked <- factor(X_train$Embarked)
    embarked_levels <- levels(X_train$Embarked)
    X_test$Embarked  <- factor(X_test$Embarked, levels = embarked_levels)
    
    d_train <- model.matrix(~ Embarked, data = X_train)[, -1, drop = FALSE]   
    d_test  <- model.matrix(~ Embarked, data = X_test)[,  -1, drop = FALSE]
    
    X_train$Embarked <- NULL
    X_test$Embarked  <- NULL
    X_train <- cbind(X_train, as.data.frame(d_train))
    X_test  <- cbind(X_test,  as.data.frame(d_test))
  }
  
  # Drop text-like columns
  to_drop <- intersect(c("Name", "Ticket", "Cabin"), colnames(X_train))
  if (length(to_drop) > 0) {
    cat(" - Dropping columns:", paste(to_drop, collapse = ", "), "\n")
    X_train <- X_train[, setdiff(colnames(X_train), to_drop), drop = FALSE]
    X_test  <- X_test[,  setdiff(colnames(X_test),  to_drop),  drop = FALSE]
  }
  

  missing_in_test <- setdiff(colnames(X_train), colnames(X_test))
  if (length(missing_in_test) > 0) {
    for (c in missing_in_test) X_test[[c]] <- 0
  }
  X_test <- X_test[, colnames(X_train), drop = FALSE]
  
  # ---------- Impute missing numeric values ----------
  cat("\n[4] Imputing NAs in numeric columns with train medians ...\n")
  num_cols <- names(Filter(is.numeric, X_train))
  for (c in num_cols) {
    med <- median(X_train[[c]], na.rm = TRUE)
    X_train[[c]][is.na(X_train[[c]])] <- med
    X_test[[c]][is.na(X_test[[c]])]   <- med
  }
  
  # ---------- Scale numeric columns ----------
  cat("\n[5] Scaling numeric features using z-score scaling ...\n")
  train_means <- sapply(X_train[, num_cols, drop = FALSE], mean)
  train_sds   <- sapply(X_train[, num_cols, drop = FALSE], sd)
  train_sds[train_sds == 0 | is.na(train_sds)] <- 1
  
  X_train[, num_cols] <- sweep(X_train[, num_cols, drop = FALSE], 2, train_means, `-`)
  X_train[, num_cols] <- sweep(X_train[, num_cols, drop = FALSE], 2, train_sds, `/`)
  X_test[,  num_cols] <- sweep(X_test[,  num_cols, drop = FALSE], 2, train_means, `-`)
  X_test[,  num_cols] <- sweep(X_test[,  num_cols, drop = FALSE], 2, train_sds, `/`)
  cat(" - Scaling complete.\n")
  
  # ---------- Train model ----------
  cat("\n[6] Training Logistic Regression ...\n")
  train_df <- cbind(Survived = y_train, X_train)
  model <- glm(Survived ~ ., data = train_df, family = binomial())
  cat(" - Model trained successfully.\n")
  
  # ---------- Evaluate ----------
  cat("\n[7] Evaluating accuracy ...\n")
  train_pred <- ifelse(predict(model, newdata = X_train, type = "response") > 0.5, 1, 0)
  test_pred  <- ifelse(predict(model, newdata = X_test,  type = "response") > 0.5, 1, 0)
  
  train_acc <- mean(train_pred == y_train, na.rm = TRUE)
  test_acc  <- mean(test_pred  == y_test,  na.rm = TRUE)
  
  cat(sprintf("Training Accuracy: %.4f\n", train_acc))
  cat(sprintf("Test Accuracy (vs gender_submission): %.4f\n", test_acc))
  
  cat("\n=== Done. ===\n")
}

if (sys.nframe() == 0) main()
