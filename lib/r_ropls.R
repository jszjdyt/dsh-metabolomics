#!/usr/bin/env Rscript
# dsh-metabolomics — ropls 官方实现脚本（PCA / PLS-DA / OPLS-DA）
#
# 调用：Rscript --vanilla lib/r_ropls.R --in <bundle> --out <out> --params '<json>'
# bundle: expr.tsv（样本×特征；首列样本名）、info.csv（样本注释，含 "_mb_group" 分组列）、
#         params.json（mode / n_comp / orth_comp / center_scale / cv_folds / permutations /
#                      seed / positive_class）
# 输出：<out>/result.json、<out>/tables/{scores,loadings,modeldf,perm}.csv、<out>/error.json
# 退出码：0 成功；3 缺前置（数据不满足）；4 运行期程序错误（上层不得据此改换方法）；
#         5 缺依赖包（转 unsupported）；其它 失败
#
# 真实实现声明：模型全部来自 ropls::opls（PCA / PLS-DA / OPLS-DA 共用该入口），
# 本脚本只负责拆包/打包，不做任何"近似重算"，也不冒充别的包。
#   * PCA      : ropls::opls(x)                       → pcaVarVn / scoreMN / loadingMN
#   * PLS-DA   : ropls::opls(x, y, orthoI = 0)        → modelDF R2X/R2Y/Q2 / vipVn
#   * OPLS-DA  : ropls::opls(x, y, predI = 1, orthoI = k)
#   * 置换检验 : ropls 自带 permI → summaryDF 的 pR2Y/pQ2 + suppLs$permMN 零分布
#   * S-plot   : p(corr) 按 ropls 自己的定义 cor(xModelMN, scoreMN)，p1 = loadingMN[, 1]
#   * 标准化   : 交给 ropls 的 scaleC（none/center/pareto/standard），不在 R 里另做
#
# 注意：ropls 的 OPLS(-DA) 只支持二分类；多分组时由 Python 侧决定"一对其余"的阳性类
# （写进 params$positive_class），本脚本只按给定的二分类标签建模，并在 result.json 里
# 记录 binary 标记与阳性类，绝不偷偷改变统计方法。

suppressPackageStartupMessages({
  if (!requireNamespace("ropls", quietly = TRUE)) {
    cat("MISSING_PACKAGE:ropls\n", file = stderr())
    cat("R 包 ropls 未安装：无法调用 opls::opls 官方实现\n", file = stderr())
    quit(save = "no", status = 5)
  }
  has_json <- requireNamespace("jsonlite", quietly = TRUE)
  suppressPackageStartupMessages(library(ropls))
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(key) {
  idx <- match(key, args)
  if (is.na(idx) || idx >= length(args)) return(NULL)
  args[idx + 1]
}

in_dir <- getarg("--in")
out_dir <- getarg("--out")
params_raw <- getarg("--params")
if (is.null(in_dir) || is.null(out_dir)) {
  cat("R 脚本缺少 --in / --out 参数\n", file = stderr())
  quit(save = "no", status = 3)
}
if (!dir.exists(in_dir)) {
  cat(paste0("bundle 目录不存在：", in_dir, "\n"), file = stderr())
  quit(save = "no", status = 3)
}
dir.create(file.path(out_dir, "tables"), recursive = TRUE, showWarnings = FALSE)

write_error <- function(code, message) {
  if (has_json) {
    jsonlite::write_json(list(code = code, message = message),
                         file.path(out_dir, "error.json"),
                         auto_unbox = TRUE, pretty = TRUE)
  }
  cat(message, "\n", file = stderr())
}
fail_prereq <- function(message) {
  write_error("prerequisite", message)
  quit(save = "no", status = 3)
}
#: 运行期程序错误：上层不得据此改换统计方法
fail_runtime <- function(message) {
  write_error("runtime", message)
  quit(save = "no", status = 4)
}

params <- list()
if (!is.null(params_raw)) {
  params <- tryCatch({
    if (has_json) jsonlite::fromJSON(params_raw, simplifyVector = TRUE) else list()
  }, error = function(e) list())
}
pick <- function(key, default) {
  value <- params[[key]]
  if (is.null(value) || length(value) == 0 || (length(value) == 1 && is.na(value))) return(default)
  value
}

mode <- tolower(as.character(pick("mode", "plsda")))
if (!(mode %in% c("pca", "plsda", "oplsda"))) {
  fail_prereq(paste0("未知的 mode：", mode, "（可选 pca / plsda / oplsda）"))
}
n_comp <- as.integer(pick("n_comp", 3))
orth_comp <- as.integer(pick("orth_comp", max(0L, n_comp - 1L)))
center_scale <- tolower(as.character(pick("center_scale", "auto")))
cv_folds <- as.integer(pick("cv_folds", 7))
permutations <- as.integer(pick("permutations", 0))
seed <- as.integer(pick("seed", 0))
positive_class <- as.character(pick("positive_class", ""))
group_col <- as.character(pick("group_col", "_mb_group"))

scale_map <- list(auto = "standard", standard = "standard", uv = "standard", zscore = "standard",
                  pareto = "pareto", par = "pareto",
                  none = "center", center = "center", centering = "center", no = "center")
scale_c <- scale_map[[center_scale]]
if (is.null(scale_c)) fail_prereq(paste0("未知的标度方式：", center_scale,
                                        "（可选 auto / pareto / none）"))

read_tsv <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.delim(path, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
}

expr <- read_tsv(file.path(in_dir, "expr.tsv"))
if (is.null(expr) || nrow(expr) == 0 || ncol(expr) == 0) {
  fail_prereq("bundle 里没有可用的 expr.tsv（样本×特征）")
}
info <- read_tsv(file.path(in_dir, "info.csv"))
if (is.null(info)) info <- data.frame(row.names = rownames(expr))

x <- as.matrix(expr)
if (!is.numeric(x)) fail_prereq("expr.tsv 含非数值列：ropls 需要纯数值矩阵")
if (any(is.na(x))) fail_prereq("expr.tsv 含缺失值：请先在 Python 侧填补（本脚本不做隐式填补）")
if (any(is.infinite(x))) fail_prereq("expr.tsv 含 Inf/-Inf")
storage.mode(x) <- "double"

n <- nrow(x)
p <- ncol(x)
if (n < 3 || p < 2) fail_prereq(sprintf("矩阵太小（%d 样本 × %d 特征），ropls 无法建模", n, p))

# 交叉验证折数：ropls 要求 crossvalI <= nrow(x)；下限 7（ropls 默认，折数太少 Q² 不稳）
cv_used <- as.integer(max(7L, min(cv_folds, n - 1L)))
if (cv_used < 7L || cv_used > (n - 1L)) cv_used <- as.integer(max(7L, n - 1L))

warnings_out <- character(0)

# ---------------------------------------------------------------------------
# 分组标签 / 二分类改造（只在 OPLS-DA 多分组时发生，且显式记录）
# ---------------------------------------------------------------------------

labels <- NULL
classes <- character(0)
binary_note <- ""
opls_binary <- FALSE

if (mode != "pca") {
  if (!(group_col %in% colnames(info))) {
    fail_prereq(paste0("info.csv 里没有分组列「", group_col, "」：ropls 判别分析需要监督标签"))
  }
  labels <- as.character(info[rownames(x), group_col])
  if (any(is.na(labels))) fail_prereq("分组列里有缺失值（NA）：请先在「样本」里补齐分组")
  classes <- unique(labels)
  if (length(classes) < 2) fail_prereq("分组只有 1 个水平：PLS-DA / OPLS-DA 至少需要 2 个分组")
  counts <- table(labels)
  if (any(counts < 2)) {
    fail_prereq(paste0("以下分组样本数不足 2：",
                       paste(names(counts)[counts < 2], collapse = "、")))
  }
  if (mode == "oplsda") {
    if (length(classes) > 2) {
      # ropls 的 OPLS-DA 只支持二分类：按 Python 侧指定的阳性类做"一对其余"
      pos <- if (!is.null(positive_class) && length(positive_class) == 1 &&
                 !is.na(positive_class) && nzchar(positive_class)) positive_class else classes[[1]]
      if (!(pos %in% classes)) {
        fail_prereq(paste0("positive_class「", pos, "」不在分组里（可选：",
                           paste(classes, collapse = "、" ), "）"))
      }
      rest_label <- "其余"
      while (rest_label %in% classes) rest_label <- paste0(rest_label, "_")
      labels_bin <- ifelse(labels == pos, pos, rest_label)
      binary_note <- sprintf(
        "多分组（%s）下 ropls 的 OPLS-DA 只支持二分类：已按一对其余构造「%s vs 其余」",
        paste(classes, collapse = "、"), pos)
      warnings_out <- c(warnings_out, binary_note)
      labels <- labels_bin
      classes <- c(pos, rest_label)
      opls_binary <- TRUE
    } else {
      opls_binary <- TRUE
    }
  }
}

# ---------------------------------------------------------------------------
# 建模（ropls 官方实现；警告收进 result，不污染 stdout）
# ---------------------------------------------------------------------------

if (mode == "pca") {
  pred_i <- max(1L, min(n_comp, n - 1L, p))
  ortho_i <- 0L
  perm_i <- 0L
} else if (mode == "oplsda") {
  pred_i <- 1L
  ortho_i <- as.integer(max(1L, min(orth_comp, min(n, p) - 1L)))
  perm_i <- as.integer(max(0L, permutations))
} else {
  pred_i <- as.integer(max(1L, min(n_comp, min(n, p) - 1L)))
  ortho_i <- 0L
  perm_i <- as.integer(max(0L, permutations))
}
if (pred_i + ortho_i > min(n, p)) {
  ortho_i <- as.integer(max(0L, min(n, p) - pred_i))
}
if (mode == "oplsda" && ortho_i < 1L) {
  fail_prereq(sprintf("正交成分数为 0（n_comp=%d，样本/特征太少），OPLS-DA 无法建模", n_comp))
}

set.seed(seed)
fit <- tryCatch(
  withCallingHandlers(
    if (mode == "pca") {
      opls(x, predI = pred_i, orthoI = 0L, crossvalI = cv_used, permI = 0L,
           scaleC = scale_c, fig.pdfC = "none", info.txtC = "none")
    } else if (mode == "oplsda") {
      opls(x, labels, predI = 1L, orthoI = ortho_i, crossvalI = cv_used,
           permI = perm_i, scaleC = scale_c, fig.pdfC = "none", info.txtC = "none")
    } else {
      opls(x, labels, predI = pred_i, orthoI = 0L, crossvalI = cv_used,
           permI = perm_i, scaleC = scale_c, fig.pdfC = "none", info.txtC = "none")
    },
    warning = function(w) {
      warnings_out <<- c(warnings_out, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      invokeRestart("muffleMessage")
    }
  ),
  error = function(e) {
    fail_runtime(paste0("ropls::opls 运行失败：", conditionMessage(e)))
  }
)

# ---------------------------------------------------------------------------
# 拆包：modelDF / summaryDF / scores / loadings / VIP / S-plot / 置换零分布
# ---------------------------------------------------------------------------

model_df <- as.data.frame(fit@modelDF, check.names = FALSE)
component_names <- rownames(model_df)
comp_type <- ifelse(grepl("^o", component_names), "ortho",
                    ifelse(grepl("^p", component_names), "predictive", "total"))

summary_df <- as.data.frame(fit@summaryDF, check.names = FALSE)
sum_pick <- function(column_names, default = NA_real_) {
  for (name in column_names) {
    if (name %in% colnames(summary_df) && nrow(summary_df) >= 1) {
      return(as.numeric(summary_df[[name]][1]))
    }
  }
  default
}
r2x_cum <- sum_pick(c("R2X(cum)"))
r2y_cum <- sum_pick(c("R2Y(cum)"))
q2_cum <- sum_pick(c("Q2(cum)"))
rmsee <- sum_pick(c("RMSEE"), NA_real_)
pre_n <- sum_pick(c("pre"), ifelse(mode == "oplsda", 1, pred_i))
ort_n <- sum_pick(c("ort"), ortho_i)
p_r2y <- sum_pick(c("pR2Y"), NA_real_)
p_q2 <- sum_pick(c("pQ2"), NA_real_)

# 逐成分表（R2X/R2X(cum)/R2Y(cum)/Q2(cum)；ropls 的 PCA 没有 R2Y/Q2 列时给 NA）
col_or_na <- function(df, name) {
  if (name %in% colnames(df)) as.numeric(df[[name]]) else rep(NA_real_, nrow(df))
}
modeldf_out <- data.frame(
  component = component_names,
  kind = comp_type,
  r2x = col_or_na(model_df, "R2X"),
  r2x_cum = col_or_na(model_df, "R2X(cum)"),
  r2y = col_or_na(model_df, "R2Y"),
  r2y_cum = col_or_na(model_df, "R2Y(cum)"),
  q2 = col_or_na(model_df, "Q2"),
  q2_cum = col_or_na(model_df, "Q2(cum)"),
  stringsAsFactors = FALSE, check.names = FALSE
)

feature_names <- colnames(x)
score_mn <- as.matrix(fit@scoreMN)
rownames(score_mn) <- rownames(x)
# 得分列名：PLS/PCA → T1..Tk；OPLS → T_pred（1 个预测成分）
if (mode == "oplsda") {
  colnames(score_mn) <- "T_pred"
} else {
  colnames(score_mn) <- paste0("T", seq_len(ncol(score_mn)))
}
if (mode == "oplsda") {
  ortho_score_mn <- as.matrix(fit@orthoScoreMN)
  rownames(ortho_score_mn) <- rownames(x)
  colnames(ortho_score_mn) <- paste0("T_orth", seq_len(ncol(ortho_score_mn)))
  scores_out <- cbind(as.data.frame(score_mn, check.names = FALSE),
                      as.data.frame(ortho_score_mn, check.names = FALSE))
  scores_out <- data.frame(sample = rownames(x), scores_out,
                           stringsAsFactors = FALSE, check.names = FALSE)
} else {
  scores_out <- data.frame(sample = rownames(x),
                           as.data.frame(score_mn, check.names = FALSE),
                           stringsAsFactors = FALSE, check.names = FALSE)
}

# 模型变量（ropls 会剔除零方差特征；xZeroVarVi 给出被剔除的列号）
x_zero_vi <- as.integer(fit@xZeroVarVi)
dropped_features <- if (length(x_zero_vi)) feature_names[x_zero_vi] else character(0)
model_features <- setdiff(feature_names, dropped_features)
if (!length(model_features)) {
  fail_prereq("ropls 建模后没有剩余特征（全部零方差）")
}

loading_mn <- as.matrix(fit@loadingMN)
pc_names <- paste0("p_corr", seq_len(ncol(loading_mn)))
p_corr_mn <- tryCatch(
  suppressWarnings(cor(as.matrix(fit@suppLs[["xModelMN"]]), as.matrix(fit@scoreMN),
                       use = "pairwise.complete.obs")),
  error = function(e) matrix(NA_real_, nrow = length(feature_names), ncol = ncol(loading_mn))
)
p_corr_mn <- as.matrix(p_corr_mn)
p_corr_names <- rownames(p_corr_mn)
if (is.null(p_corr_names) || length(p_corr_names) != nrow(p_corr_mn)) {
  p_corr_names <- feature_names[seq_len(nrow(p_corr_mn))]
}
colnames(p_corr_mn) <- pc_names
p_names <- paste0("loading_p", seq_len(ncol(loading_mn)))
loadings_out <- data.frame(feature = model_features,
                           stringsAsFactors = FALSE, check.names = FALSE)
for (j in seq_len(ncol(loading_mn))) {
  loadings_out[[p_names[j]]] <- as.numeric(loading_mn[model_features, j])
}
for (j in seq_len(ncol(p_corr_mn))) {
  values <- rep(NA_real_, length(model_features))
  hit <- match(model_features, p_corr_names)
  ok <- !is.na(hit)
  values[ok] <- as.numeric(p_corr_mn[hit[ok], j])
  loadings_out[[pc_names[j]]] <- values
}
loadings_out[["vip"]] <- as.numeric(fit@vipVn[model_features])
if (mode == "oplsda") {
  ortho_loading_mn <- as.matrix(fit@orthoLoadingMN)
  for (j in seq_len(ncol(ortho_loading_mn))) {
    loadings_out[[paste0("ortho_loading_o", j)]] <-
      as.numeric(ortho_loading_mn[model_features, j])
  }
  loadings_out[["ortho_vip"]] <- as.numeric(fit@orthoVipVn[model_features])
}

# PCA：解释方差（ropls 的 pcaVarVn；与 SVD 的 var = s²/(n-1) 语义一致）。
# pcaVarVn 只给 predI 个成分，所以另给总方差（模型标度后矩阵的方差之和），
# Python 侧用 pca_var / pca_total_var 得到解释率，避免只用前几个成分当总量。
pca_var <- NULL
pca_total_var <- NULL
if (mode == "pca") {
  pca_var <- as.numeric(fit@pcaVarVn)
  x_model_mn <- as.matrix(fit@suppLs[["xModelMN"]])
  variances <- apply(x_model_mn, 2, stats::var)
  variances <- variances[is.finite(variances)]
  pca_total_var <- if (length(variances)) sum(variances) else NA_real_
  if (!length(pca_var) || !is.finite(sum(pca_var)) || sum(pca_var) <= 0) {
    pca_var <- rep(NA_real_, ncol(loading_mn))
  }
}

# 置换检验零分布（ropls 自带 permI；第 1 行是观测模型，sim=1，其余是打乱标签的重拟合）
perm_out <- NULL
perm_info <- list(n_perms = 0L, r2y_obs = r2y_cum, q2_obs = q2_cum,
                  p_r2y = p_r2y, p_q2 = p_q2)
perm_mn <- fit@suppLs[["permMN"]]
if (!is.null(perm_mn) && is.matrix(perm_mn) && nrow(perm_mn) > 1) {
  perm_df <- as.data.frame(perm_mn, check.names = FALSE)
  perm_df <- data.frame(perm_index = seq_len(nrow(perm_df)), perm_df,
                        stringsAsFactors = FALSE, check.names = FALSE)
  perm_out <- perm_df
  perm_info$n_perms <- as.integer(nrow(perm_mn) - 1L)
  if ("R2Y(cum)" %in% colnames(perm_mn)) perm_info$r2y_obs <- as.numeric(perm_mn[1, "R2Y(cum)"])
  if ("Q2(cum)" %in% colnames(perm_mn)) perm_info$q2_obs <- as.numeric(perm_mn[1, "Q2(cum)"])
}

# ---------------------------------------------------------------------------
# 落盘（表用 row.names 写，Python 侧 index_col=0 读回，特征/样本名不被改写）
# ---------------------------------------------------------------------------

write_table <- function(frame, name) {
  path <- file.path(out_dir, "tables", paste0(name, ".csv"))
  write.csv(frame, path, row.names = FALSE)
  invisible(path)
}
write_indexed <- function(frame, index, name) {
  path <- file.path(out_dir, "tables", paste0(name, ".csv"))
  write.csv(frame, path, row.names = index)
  invisible(path)
}

write_indexed(scores_out[, setdiff(colnames(scores_out), "sample"), drop = FALSE],
              scores_out$sample, "scores")
loadings_table <- loadings_out
rownames(loadings_table) <- loadings_table$feature
loadings_table$feature <- NULL
write_indexed(loadings_table, rownames(loadings_table), "loadings")
write_indexed(modeldf_out, modeldf_out$component, "modeldf")
if (!is.null(perm_out)) {
  write_indexed(perm_out, perm_out$perm_index, "perm")
}

model_type <- fit@typeC
result <- list(
  engine = "ropls",
  model_type = model_type,
  mode = mode,
  method_call = "ropls::opls",
  ropls_version = as.character(packageVersion("ropls")),
  r_version = as.character(getRversion()),
  scaleC = scale_c,
  predI = pred_i,
  orthoI = ortho_i,
  n_comp = as.integer(pre_n),
  n_orth = as.integer(ort_n),
  crossvalI = cv_used,
  permI = if (is.null(perm_out)) 0L else as.integer(perm_info$n_perms),
  n_samples = as.integer(n),
  n_features_input = as.integer(p),
  n_features_model = as.integer(length(model_features)),
  n_features_zero_var = as.integer(length(dropped_features)),
  zero_var_features = I(as.character(dropped_features)),
  classes = I(as.character(classes)),
  labels = I(as.character(if (is.null(labels)) rep("", n) else labels)),
  binary = isTRUE(opls_binary),
  positive_class = as.character(if (nzchar(positive_class)) positive_class else ""),
  summary = list(
    r2x_cum = r2x_cum, r2y_cum = r2y_cum, q2_cum = q2_cum, rmsee = rmsee,
    pre = as.integer(pre_n), ort = as.integer(ort_n),
    p_r2y = p_r2y, p_q2 = p_q2
  ),
  perm = perm_info,
  pca_var = if (is.null(pca_var)) NULL else I(pca_var),
  pca_total_var = if (is.null(pca_total_var)) NULL else pca_total_var,
  warnings = I(as.character(unique(warnings_out))),
  notes = I(as.character(c(
    sprintf("模型由 %s %s 计算（%s）", "ropls", as.character(packageVersion("ropls")),
            model_type),
    binary_note
  )))
)

if (has_json) {
  jsonlite::write_json(result, file.path(out_dir, "result.json"),
                       auto_unbox = TRUE, pretty = TRUE, null = "null", digits = 12)
} else {
  writeLines(
    sprintf(paste0('{"engine":"ropls","model_type":"%s","summary":{"r2x_cum":%s,',
                   '"r2y_cum":%s,"q2_cum":%s},"warnings":[]}'),
            model_type,
            ifelse(is.na(r2x_cum), "null", format(r2x_cum, digits = 10)),
            ifelse(is.na(r2y_cum), "null", format(r2y_cum, digits = 10)),
            ifelse(is.na(q2_cum), "null", format(q2_cum, digits = 10))),
    file.path(out_dir, "result.json")
  )
}

quit(save = "no", status = 0)
