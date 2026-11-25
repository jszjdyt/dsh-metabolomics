"""
dsh-metabolomics — 通路富集与化学分类富集（DESIGN.md §3 step enrich / §4 group=通路与化学分类 / §5 图）

本模块提供：
  STEPS   : enrich（method=mummichog | ora_kegg）
  ANALYSES: mummichog_pathway / ora_kegg / ora_hmdb_class / metabolite_set_enrich
  PLOTS   : enrich_bar / enrich_dot

方法（全部内置实现；本机没有 xcms / pyopenms / MetaboAnalystR / mummichog 官方包）：
  * Mummichog 式质量映射：把差异显著特征的 m/z 与"通路—理论离子质量"表按 ppm 容差匹配，
    得到每条通路的命中数，再用"随机抽取同样数量的特征"做置换得到经验 p 值（BH 校正）。
    依据：Li et al. 2013 (mummichog, PLoS Comput Biol)；Uppal et al. 2017 (xref 代谢组标准化)；
    Sumner et al. 2007（代谢组学质谱精度准则）。
  * 理论离子质量不依赖外部库：先按内置代谢物分子式算单同位素质量（mb_common.formula_mass），
    再用 mb_common.ADDUCTS 的质子/钠/氯等加合物换算成理论离子 m/z。
  * ORA（过表达分析）：超几何检验（Fisher 精确检验的单尾形式）+ BH。
    依据：Kanehisa et al. 2023 (KEGG)；Wishart et al. 2022 (HMDB 5.0 化学分类)。
  * MSEA（代谢物集富集）：按 log2FC 排序做加权 KS 游走（GSEA 思想），置换零分布 + BH。
    依据：Subramanian et al. 2005 (GSEA)；Xia & Wishart 2015 (MetaboAnalyst MSEA)。

数据来源优先级（通路的"成员集合"）：
  1) `<workdir>/genesets/*.json` 用户/在线缓存（支持通路→成员名列表、通路→KEGG 编号列表、
     以及成员为 {"name","mass"} 的对象——后者会把中性质量并入理论质量表）；
  2) 内置库 MB_PATHWAYS（26 条 KEGG 通路，每条 5–20 个代谢物名）；
  3) 可选在线补充 rest.kegg.jp（params online=true 才联网；失败只 note，不阻塞）。
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from mb_common import (
    ADDUCTS,
    NEGATIVE_ADDUCTS,
    PALETTE,
    POSITIVE_ADDUCTS,
    bad_request,
    bh_fdr,
    data_file,
    export_csv,
    formula_mass,
    hmdb_class_of,
    new_fig,
    prerequisite,
    result,
    save_fig,
    table,
)

SEED = 0
DEFAULT_PPM = 10.0
DEFAULT_P_CUT = 0.05
DEFAULT_N_PERM = 1000
DEFAULT_MIN_SIZE = 3
MAX_SIG = 2000          # 置换检验的特征上限（超过按 |log2FC| 取前 2000）
MAX_HIT_LABELS = 12     # hits 列最多列几个特征

# ---------------------------------------------------------------------------
# 1. 内置代谢物（分子式 → 中性单同位素质量）与 KEGG 通路库
#    质量一律由 mb_common.formula_mass() 从分子式计算，不在代码里硬编码 Da 值；
#    分子式取自 HMDB 5.0 / KEGG COMPOUND 的常规中性式（Wishart 2022；Kanehisa 2023）。
# ---------------------------------------------------------------------------

MB_METABOLITES: dict[str, str] = {
    # --- 糖 / 磷酸糖 ---
    "D-Glucose": "C6H12O6",
    "D-Fructose": "C6H12O6",
    "Glucose 6-phosphate": "C6H13O9P",
    "Fructose 6-phosphate": "C6H13O9P",
    "Fructose 1,6-bisphosphate": "C6H14O12P2",
    "Glyceraldehyde 3-phosphate": "C3H7O6P",
    "Dihydroxyacetone phosphate": "C3H7O6P",
    "3-Phospho-D-glycerate": "C3H7O7P",
    "2-Phospho-D-glycerate": "C3H7O7P",
    "Phosphoenolpyruvate": "C3H5O6P",
    "6-Phospho-D-glucono-1,5-lactone": "C6H11O9P",
    "6-Phospho-D-gluconate": "C6H13O10P",
    "D-Ribulose 5-phosphate": "C5H11O8P",
    "D-Ribose 5-phosphate": "C5H11O8P",
    "D-Xylulose 5-phosphate": "C5H11O8P",
    "Sedoheptulose 7-phosphate": "C7H15O10P",
    "D-Erythrose 4-phosphate": "C4H9O7P",
    "D-Ribose": "C5H10O5",
    "D-Xylose": "C5H10O5",
    "Ribitol": "C5H12O5",
    # --- 有机酸 / TCA ---
    "Pyruvate": "C3H4O3",
    "Lactate": "C3H6O3",
    "Citrate": "C6H8O7",
    "Isocitrate": "C6H8O7",
    "cis-Aconitate": "C6H6O6",
    "2-Oxoglutarate": "C5H6O5",
    "Succinate": "C4H6O4",
    "Fumarate": "C4H4O4",
    "L-Malate": "C4H6O5",
    "Oxaloacetate": "C4H4O5",
    "Succinyl-CoA": "C25H40N7O19P3S",
    "Acetyl-CoA": "C23H38N7O17P3S",
    "Malonyl-CoA": "C24H38N7O19P3S",
    "Acetate": "C2H4O2",
    "Formate": "CH2O2",
    "Acetaldehyde": "C2H4O",
    "Ethanol": "C2H6O",
    "Methylglyoxal": "C3H4O2",
    "Glutarate": "C5H8O4",
    "3-Hydroxyglutarate": "C5H8O5",
    # --- 核苷酸（嘌呤 / 嘧啶） ---
    "Adenine": "C5H5N5",
    "Adenosine": "C10H13N5O4",
    "AMP": "C10H14N5O7P",
    "ADP": "C10H15N5O10P2",
    "ATP": "C10H16N5O13P3",
    "Guanine": "C5H5N5O",
    "Guanosine": "C10H13N5O5",
    "GMP": "C10H14N5O8P",
    "GDP": "C10H15N5O11P2",
    "GTP": "C10H16N5O14P3",
    "Inosine": "C10H12N4O5",
    "Hypoxanthine": "C5H4N4O",
    "Xanthine": "C5H4N4O2",
    "Uric acid": "C5H4N4O3",
    "IMP": "C10H13N4O8P",
    "Allantoin": "C4H6N4O3",
    "Xanthosine": "C10H12N4O6",
    "Cytosine": "C4H5N3O",
    "Uracil": "C4H4N2O2",
    "Thymine": "C5H6N2O2",
    "Cytidine": "C9H13N3O5",
    "Uridine": "C9H12N2O6",
    "Thymidine": "C10H14N2O5",
    "CMP": "C9H14N3O8P",
    "UMP": "C9H13N2O9P",
    "dTMP": "C10H15N2O8P",
    "Orotate": "C5H4N2O4",
    "Carbamoyl phosphate": "CH4NO5P",
    "beta-Alanine": "C3H7NO2",
    # --- 氨基酸（Aminoacyl-tRNA biosynthesis 的 20 种标准氨基酸） ---
    "Glycine": "C2H5NO2",
    "L-Alanine": "C3H7NO2",
    "L-Serine": "C3H7NO3",
    "L-Proline": "C5H9NO2",
    "L-Valine": "C5H11NO2",
    "L-Threonine": "C4H9NO3",
    "L-Leucine": "C6H13NO2",
    "L-Isoleucine": "C6H13NO2",
    "L-Asparagine": "C4H8N2O3",
    "L-Aspartate": "C4H7NO4",
    "L-Glutamine": "C5H10N2O3",
    "L-Lysine": "C6H14N2O2",
    "L-Glutamate": "C5H9NO4",
    "L-Methionine": "C5H11NO2S",
    "L-Histidine": "C6H9N3O2",
    "L-Phenylalanine": "C9H11NO2",
    "L-Arginine": "C6H14N4O2",
    "L-Tyrosine": "C9H11NO3",
    "L-Tryptophan": "C11H12N2O2",
    "L-Cysteine": "C3H7NO2S",
    # --- 胆汁酸 ---
    "Cholic acid": "C24H40O5",
    "Chenodeoxycholic acid": "C24H40O4",
    "Deoxycholic acid": "C24H40O4",
    "Lithocholic acid": "C24H40O3",
    "Glycocholic acid": "C26H43NO6",
    "Taurocholic acid": "C26H45NO7S",
    "Glycochenodeoxycholic acid": "C26H43NO5",
    "Taurochenodeoxycholic acid": "C26H45NO6S",
    # --- 脂肪酸 / 脂质 ---
    "Palmitic acid": "C16H32O2",
    "Stearic acid": "C18H36O2",
    "Oleic acid": "C18H34O2",
    "Linoleic acid": "C18H32O2",
    "Arachidonic acid": "C20H32O2",
    "Myristic acid": "C14H28O2",
    "Lauric acid": "C12H24O2",
    "Decanoic acid": "C10H20O2",
    "Hexanoic acid": "C6H12O2",
    "Butyric acid": "C4H8O2",
    "Carnitine": "C7H15NO3",
    "Palmitoylcarnitine": "C23H45NO4",
    "Sphingosine": "C18H37NO2",
    "Sphinganine": "C18H39NO2",
    "Sphingosine 1-phosphate": "C18H38NO5P",
    "Ceramide (d18:1/16:0)": "C34H67NO3",
    "Glucosylceramide (d18:1/16:0)": "C40H77NO8",
    "Sphingomyelin (d18:1/16:0)": "C39H79N2O6P",
    "Psychosine": "C24H47NO7",
    "Cholesterol": "C27H46O",
    "7-Dehydrocholesterol": "C27H44O",
    "7-alpha-Hydroxycholesterol": "C27H46O2",
    "Desmosterol": "C27H44O",
    "Lathosterol": "C27H46O",
    "Lanosterol": "C30H50O",
    "Squalene": "C30H50",
    "Squalene 2,3-epoxide": "C30H50O",
    "Mevalonate": "C6H12O4",
    "Farnesyl pyrophosphate": "C15H28O7P2",
    "Isopentenyl pyrophosphate": "C5H12O7P2",
    # --- 甘油磷脂 ---
    "Phosphocholine": "C5H14NO4P",
    "Glycerophosphocholine": "C8H20NO6P",
    "Phosphoethanolamine": "C2H8NO4P",
    "Glycerophosphoethanolamine": "C5H14NO6P",
    "Ethanolamine": "C2H7NO",
    "Glycerol 3-phosphate": "C3H9O6P",
    "CDP-choline": "C14H26N4O11P2",
    "Phosphatidylcholine (34:1)": "C42H82NO8P",
    "Phosphatidylethanolamine (34:1)": "C39H76NO8P",
    # --- 精氨酸 / 脯氨酸 ---
    "L-Ornithine": "C5H12N2O2",
    "L-Citrulline": "C6H13N3O3",
    "Urea": "CH4N2O",
    "trans-4-Hydroxy-L-proline": "C5H9NO3",
    "Putrescine": "C4H12N2",
    "Spermidine": "C7H19N3",
    "Spermine": "C10H26N4",
    "Agmatine": "C5H14N4",
    "Creatine": "C4H9N3O2",
    "Creatinine": "C4H7N3O",
    "N-Acetyl-L-glutamate": "C7H11NO5",
    "L-Argininosuccinate": "C10H18N4O6",
    "1-Pyrroline-5-carboxylate": "C5H7NO2",
    # --- 色氨酸 ---
    "Serotonin": "C10H12N2O",
    "5-Hydroxyindoleacetate": "C10H9NO3",
    "L-Kynurenine": "C10H12N2O3",
    "Kynurenic acid": "C10H7NO3",
    "Quinolinic acid": "C7H5NO4",
    "Indole": "C8H7N",
    "Indole-3-acetate": "C10H9NO2",
    "Indole-3-lactate": "C11H11NO3",
    "Tryptamine": "C10H12N2",
    "3-Hydroxykynurenine": "C10H12N2O4",
    "Anthranilic acid": "C7H7NO2",
    "5-Hydroxy-L-tryptophan": "C11H12N2O3",
    "Melatonin": "C13H16N2O2",
    "N-Formylkynurenine": "C11H12N2O4",
    # --- 半胱氨酸 / 甲硫氨酸 ---
    "L-Cystine": "C6H12N2O4S2",
    "L-Homocysteine": "C4H9NO2S",
    "Taurine": "C2H7NO3S",
    "Glutathione": "C10H17N3O6S",
    "Glutathione disulfide": "C20H32N6O12S2",
    "Cysteine sulfinate": "C3H7NO4S",
    "S-Adenosyl-L-methionine": "C15H22N6O5S",
    "L-Cystathionine": "C7H14N2O4S",
    "Hypotaurine": "C2H7NO2S",
    "Methionine sulfoxide": "C5H11NO3S",
    "Dimethylglycine": "C4H9NO2",
    "Sarcosine": "C3H7NO2",
    "gamma-L-Glutamyl-L-cysteine": "C8H14N2O5S",
    "5-Oxoproline": "C5H7NO3",
    "Cysteinylglycine": "C5H10N2O3S",
    # --- 烟酸 / 烟酰胺 ---
    "Nicotinate": "C6H5NO2",
    "Nicotinamide": "C6H6N2O",
    "Nicotinamide N-oxide": "C6H6N2O2",
    "Trigonelline": "C7H7NO2",
    "N1-Methyl-2-pyridone-5-carboxamide": "C7H8N2O2",
    "NAD": "C21H27N7O14P2",
    "NADH": "C21H29N7O14P2",
    "NADP": "C21H28N7O17P3",
    "NADPH": "C21H30N7O17P3",
    # --- 氧化磷酸化 ---
    "Phosphate": "H3O4P",
    "Pyrophosphate": "H4O7P2",
    "FAD": "C27H33N9O15P2",
    "FMN": "C17H21N4O9P",
    "Ubiquinone-10": "C59H90O4",
    "Creatine phosphate": "C4H10N3O5P",
    # --- 组氨酸 ---
    "Histamine": "C5H9N3",
    "1-Methylhistidine": "C7H11N3O2",
    "3-Methylhistidine": "C7H11N3O2",
    "Imidazole-4-acetate": "C5H6N2O2",
    "Urocanate": "C6H6N2O2",
    "N-Acetyl-L-histidine": "C8H11N3O3",
    "L-Carnosine": "C9H14N4O3",
    "L-Anserine": "C10H16N4O3",
    "Formiminoglutamate": "C6H10N2O4",
    # --- 酪氨酸 ---
    "Dopamine": "C8H11NO2",
    "L-DOPA": "C9H11NO4",
    "Norepinephrine": "C8H11NO3",
    "Epinephrine": "C9H13NO3",
    "Homovanillate": "C9H10O4",
    "Vanillylmandelate": "C9H10O5",
    "4-Hydroxyphenylpyruvate": "C9H8O4",
    "4-Hydroxyphenyllactate": "C9H10O4",
    "Tyramine": "C8H11NO",
    "Gentisate": "C7H6O4",
    "3,4-Dihydroxyphenylacetate": "C8H8O4",
    # --- 苯丙氨酸 ---
    "Phenylpyruvate": "C9H8O3",
    "Phenyllactate": "C9H10O3",
    "Phenylacetate": "C8H8O2",
    "Phenylacetylglutamine": "C13H16N2O4",
    "Benzoate": "C7H6O2",
    "Hippurate": "C9H9NO3",
    "4-Hydroxyphenylacetate": "C8H8O3",
    "Mandelate": "C8H8O3",
    "trans-Cinnamate": "C9H8O2",
    # --- 赖氨酸降解 ---
    "Saccharopine": "C11H20N2O5",
    "alpha-Aminoadipate": "C6H11NO4",
    "2-Aminoadipate 6-semialdehyde": "C6H11NO3",
    "Pipecolate": "C6H11NO2",
    "Cadaverine": "C5H14N2",
    "5-Aminopentanoate": "C5H11NO2",
    "N6-Acetyl-L-lysine": "C8H16N2O2",
    "N6,N6,N6-Trimethyl-L-lysine": "C9H20N2O2",
    # --- 肌醇磷酸 ---
    "myo-Inositol": "C6H12O6",
    "myo-Inositol 1-phosphate": "C6H13O9P",
    "myo-Inositol 4-phosphate": "C6H13O9P",
    "1D-myo-Inositol 1,4,5-trisphosphate": "C6H15O15P3",
    "D-Glucuronate": "C6H10O7",
    "D-Glucarate": "C6H10O8",
    # --- 叶酸 ---
    "Folate": "C19H19N7O6",
    "Dihydrofolate": "C19H21N7O6",
    "Tetrahydrofolate": "C19H23N7O6",
    "5-Methyltetrahydrofolate": "C20H25N7O6",
    "5,10-Methylenetetrahydrofolate": "C20H23N7O6",
    "10-Formyltetrahydrofolate": "C20H23N7O7",
    "5-Formyltetrahydrofolate": "C20H23N7O7",
    "Pterin": "C6H5N5O",
    "4-Aminobenzoate": "C7H7NO2",
    # --- 视黄醇 ---
    "Retinol": "C20H30O",
    "Retinal": "C20H28O",
    "Retinoic acid": "C20H28O2",
    "13-cis-Retinoic acid": "C20H28O2",
    "4-Oxoretinol": "C20H28O2",
    "Retinyl palmitate": "C36H60O2",
    "Retinyl acetate": "C22H32O2",
    "beta-Carotene": "C40H56",
    "alpha-Carotene": "C40H56",
    # --- 咖啡因 ---
    "Caffeine": "C8H10N4O2",
    "Theobromine": "C7H8N4O2",
    "Theophylline": "C7H8N4O2",
    "Paraxanthine": "C7H8N4O2",
    "1-Methylxanthine": "C6H6N4O2",
    "3-Methylxanthine": "C6H6N4O2",
    "7-Methylxanthine": "C6H6N4O2",
    "1,3-Dimethylurate": "C7H8N4O3",
    "1-Methylurate": "C6H6N4O3",
    # --- 类固醇激素 ---
    "Pregnenolone": "C21H32O2",
    "Progesterone": "C21H30O2",
    "Cortisol": "C21H30O5",
    "Corticosterone": "C21H30O4",
    "Aldosterone": "C21H28O5",
    "Testosterone": "C19H28O2",
    "Androstenedione": "C19H26O2",
    "Dehydroepiandrosterone": "C19H28O2",
    "17beta-Estradiol": "C18H24O2",
    "Estrone": "C18H22O2",
    "17alpha-Hydroxyprogesterone": "C21H30O3",
    "11-Deoxycortisol": "C21H30O4",
}

#: 内置 KEGG 通路库：通路名 → 代谢物名（每条 5–25 个；DESIGN §4 group=通路与化学分类）
MB_PATHWAYS: dict[str, tuple[str, ...]] = {
    "Glycolysis / Gluconeogenesis": (
        "D-Glucose", "D-Fructose", "Glucose 6-phosphate", "Fructose 6-phosphate",
        "Fructose 1,6-bisphosphate", "Glyceraldehyde 3-phosphate", "Dihydroxyacetone phosphate",
        "3-Phospho-D-glycerate", "2-Phospho-D-glycerate", "Phosphoenolpyruvate",
        "Pyruvate", "Lactate", "Acetyl-CoA", "Ethanol",
    ),
    "Citrate cycle (TCA cycle)": (
        "Citrate", "Isocitrate", "cis-Aconitate", "2-Oxoglutarate", "Succinate",
        "Fumarate", "L-Malate", "Oxaloacetate", "Succinyl-CoA", "Acetyl-CoA", "Pyruvate",
    ),
    "Pyruvate metabolism": (
        "Pyruvate", "Lactate", "Acetyl-CoA", "Malonyl-CoA", "L-Malate", "Oxaloacetate",
        "Acetaldehyde", "Ethanol", "Acetate", "Formate", "Methylglyoxal",
    ),
    "Purine metabolism": (
        "Adenine", "Adenosine", "AMP", "ADP", "ATP", "Guanine", "Guanosine", "GMP",
        "GDP", "GTP", "Inosine", "Hypoxanthine", "Xanthine", "Uric acid", "IMP",
        "Allantoin", "Xanthosine",
    ),
    "Pyrimidine metabolism": (
        "Cytosine", "Uracil", "Thymine", "Cytidine", "Uridine", "Thymidine", "CMP",
        "UMP", "dTMP", "Orotate", "Carbamoyl phosphate", "beta-Alanine",
    ),
    "Aminoacyl-tRNA biosynthesis": (
        "Glycine", "L-Alanine", "L-Serine", "L-Proline", "L-Valine", "L-Threonine",
        "L-Leucine", "L-Isoleucine", "L-Asparagine", "L-Aspartate", "L-Glutamine",
        "L-Lysine", "L-Glutamate", "L-Methionine", "L-Histidine", "L-Phenylalanine",
        "L-Arginine", "L-Tyrosine", "L-Tryptophan", "L-Cysteine",
    ),
    "Bile acid biosynthesis": (
        "Cholic acid", "Chenodeoxycholic acid", "Deoxycholic acid", "Lithocholic acid",
        "Glycocholic acid", "Taurocholic acid", "Glycochenodeoxycholic acid",
        "Taurochenodeoxycholic acid", "Cholesterol", "7-alpha-Hydroxycholesterol",
    ),
    "Fatty acid metabolism": (
        "Palmitic acid", "Stearic acid", "Oleic acid", "Linoleic acid", "Arachidonic acid",
        "Myristic acid", "Lauric acid", "Decanoic acid", "Hexanoic acid", "Butyric acid",
        "Malonyl-CoA", "Acetyl-CoA", "Carnitine", "Palmitoylcarnitine",
    ),
    "Sphingolipid metabolism": (
        "Sphingosine", "Sphinganine", "Sphingosine 1-phosphate", "Ceramide (d18:1/16:0)",
        "Glucosylceramide (d18:1/16:0)", "Sphingomyelin (d18:1/16:0)", "Psychosine",
        "L-Serine", "Palmitic acid",
    ),
    "Glycerophospholipid metabolism": (
        "Phosphocholine", "Glycerophosphocholine", "Phosphoethanolamine",
        "Glycerophosphoethanolamine", "Ethanolamine", "Glycerol 3-phosphate",
        "CDP-choline", "Phosphatidylcholine (34:1)", "Phosphatidylethanolamine (34:1)",
        "Arachidonic acid", "Carnitine",
    ),
    "Steroid biosynthesis": (
        "Cholesterol", "7-Dehydrocholesterol", "7-alpha-Hydroxycholesterol", "Desmosterol",
        "Lathosterol", "Lanosterol", "Squalene", "Squalene 2,3-epoxide", "Mevalonate",
        "Farnesyl pyrophosphate", "Isopentenyl pyrophosphate", "Pregnenolone",
    ),
    "Steroid hormone biosynthesis": (
        "Pregnenolone", "Progesterone", "Cortisol", "Corticosterone", "Aldosterone",
        "Testosterone", "Androstenedione", "Dehydroepiandrosterone", "17beta-Estradiol",
        "Estrone", "17alpha-Hydroxyprogesterone", "11-Deoxycortisol", "Cholesterol",
    ),
    "Arginine and proline metabolism": (
        "L-Arginine", "L-Ornithine", "L-Citrulline", "Urea", "L-Proline",
        "trans-4-Hydroxy-L-proline", "Putrescine", "Spermidine", "Spermine", "Agmatine",
        "Creatine", "Creatinine", "N-Acetyl-L-glutamate", "L-Argininosuccinate",
        "1-Pyrroline-5-carboxylate", "L-Glutamate",
    ),
    "Tryptophan metabolism": (
        "L-Tryptophan", "Serotonin", "5-Hydroxyindoleacetate", "L-Kynurenine",
        "Kynurenic acid", "Quinolinic acid", "Indole", "Indole-3-acetate",
        "Indole-3-lactate", "Tryptamine", "3-Hydroxykynurenine", "Anthranilic acid",
        "5-Hydroxy-L-tryptophan", "Melatonin", "N-Formylkynurenine",
    ),
    "Cysteine and methionine metabolism": (
        "L-Cysteine", "L-Methionine", "L-Cystine", "L-Homocysteine", "Taurine",
        "Glutathione", "Cysteine sulfinate", "S-Adenosyl-L-methionine",
        "L-Cystathionine", "Hypotaurine", "Methionine sulfoxide", "Dimethylglycine",
        "Sarcosine", "L-Serine",
    ),
    "Glutathione metabolism": (
        "Glutathione", "Glutathione disulfide", "L-Cysteine", "Glycine", "L-Glutamate",
        "gamma-L-Glutamyl-L-cysteine", "5-Oxoproline", "Cysteinylglycine", "L-Ornithine",
        "Spermidine", "Putrescine",
    ),
    "Histidine metabolism": (
        "L-Histidine", "Histamine", "1-Methylhistidine", "3-Methylhistidine",
        "Imidazole-4-acetate", "Urocanate", "N-Acetyl-L-histidine", "L-Carnosine",
        "L-Anserine", "Formiminoglutamate", "beta-Alanine",
    ),
    "Tyrosine metabolism": (
        "L-Tyrosine", "Dopamine", "L-DOPA", "Norepinephrine", "Epinephrine",
        "Homovanillate", "Vanillylmandelate", "4-Hydroxyphenylpyruvate",
        "4-Hydroxyphenyllactate", "Tyramine", "Gentisate", "3,4-Dihydroxyphenylacetate",
    ),
    "Phenylalanine metabolism": (
        "L-Phenylalanine", "Phenylpyruvate", "Phenyllactate", "Phenylacetate",
        "Phenylacetylglutamine", "Benzoate", "Hippurate", "L-Tyrosine",
        "4-Hydroxyphenylacetate", "Mandelate", "trans-Cinnamate",
    ),
    "Lysine degradation": (
        "L-Lysine", "Saccharopine", "alpha-Aminoadipate", "2-Aminoadipate 6-semialdehyde",
        "Pipecolate", "Cadaverine", "5-Aminopentanoate", "Glutarate",
        "N6-Acetyl-L-lysine", "N6,N6,N6-Trimethyl-L-lysine", "Carnitine",
        "3-Hydroxyglutarate",
    ),
    "Pentose phosphate pathway": (
        "D-Glucose", "Glucose 6-phosphate", "6-Phospho-D-glucono-1,5-lactone",
        "6-Phospho-D-gluconate", "D-Ribulose 5-phosphate", "D-Ribose 5-phosphate",
        "D-Xylulose 5-phosphate", "Sedoheptulose 7-phosphate", "D-Erythrose 4-phosphate",
        "Fructose 6-phosphate", "Glyceraldehyde 3-phosphate", "D-Ribose", "D-Xylose",
        "Ribitol",
    ),
    "Inositol phosphate metabolism": (
        "myo-Inositol", "myo-Inositol 1-phosphate", "myo-Inositol 4-phosphate",
        "1D-myo-Inositol 1,4,5-trisphosphate", "D-Glucuronate", "D-Glucarate",
        "Glycerol 3-phosphate", "Glucose 6-phosphate",
    ),
    "Folate biosynthesis": (
        "Folate", "Dihydrofolate", "Tetrahydrofolate", "5-Methyltetrahydrofolate",
        "5,10-Methylenetetrahydrofolate", "10-Formyltetrahydrofolate",
        "5-Formyltetrahydrofolate", "Pterin", "4-Aminobenzoate", "L-Glutamate",
    ),
    "Nicotinate and nicotinamide metabolism": (
        "Nicotinate", "Nicotinamide", "Nicotinamide N-oxide", "Trigonelline",
        "N1-Methyl-2-pyridone-5-carboxamide", "NAD", "NADH", "NADP", "NADPH",
        "Quinolinic acid",
    ),
    "Oxidative phosphorylation": (
        "ATP", "ADP", "AMP", "GTP", "GDP", "Phosphate", "Pyrophosphate", "NADH",
        "NADPH", "FAD", "FMN", "Ubiquinone-10", "Creatine phosphate", "Succinate",
    ),
    "Retinol metabolism": (
        "Retinol", "Retinal", "Retinoic acid", "13-cis-Retinoic acid", "4-Oxoretinol",
        "Retinyl palmitate", "Retinyl acetate", "beta-Carotene", "alpha-Carotene",
    ),
    "Caffeine metabolism": (
        "Caffeine", "Theobromine", "Theophylline", "Paraxanthine", "1-Methylxanthine",
        "3-Methylxanthine", "7-Methylxanthine", "Xanthine", "Uric acid",
        "1,3-Dimethylurate", "1-Methylurate",
    ),
}


def _build_library() -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """把分子式表换算成中性单同位素质量，并丢掉库里缺分子式的成员。"""
    masses: dict[str, float] = {}
    for name, formula in MB_METABOLITES.items():
        try:
            value = float(formula_mass(formula))
        except Exception:  # noqa: BLE001 - 分子式写错不应拖垮整个插件
            continue
        if math.isfinite(value) and value > 0:
            masses[name] = value
    pathways: dict[str, tuple[str, ...]] = {}
    for path, members in MB_PATHWAYS.items():
        keep = tuple(dict.fromkeys(m for m in members if m in masses))
        if keep:
            pathways[str(path)] = keep
    return masses, pathways


#: 代谢物名 → 中性单同位素质量（Da）
MB_METABOLITE_MASS, MB_PATHWAY_MEMBERS = _build_library()


# ---------------------------------------------------------------------------
# 2. 名称归一化 / 注释 → 通路、化学分类
# ---------------------------------------------------------------------------

_NAME_SPLIT = re.compile(r"[;,|/]+")
_KEGG_ID = re.compile(r"^[A-Za-z]\d{3,6}$")
_ACID_SUFFIXES = ("ic acid", "icacid", "ate", "acid")


def _norm_name(text) -> str:
    """代谢物名归一：去盐/立体化学前缀 + 统一 "…ic acid" / "…ate" 词干，便于名称匹配。

    例：'L-Lactic acid' / 'Lactate' → 'lact'；'Glutamic acid' / 'Glutamate' → 'glutam'。
    """
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"\([^)]*\)", " ", raw)          # 去掉括注
    raw = raw.replace("α", "alpha").replace("β", "beta")
    raw = re.sub(r"^(l|d|dl|cis|trans)[\s\-]+", "", raw.strip())
    raw = re.sub(r"[^a-z0-9]+", " ", raw).strip()
    compact = raw.replace(" ", "")
    for suffix in _ACID_SUFFIXES:
        if compact.endswith(suffix) and len(compact) > len(suffix) + 2:
            return compact[: -len(suffix)]
    return compact


def _feature_list(ctx) -> list[str]:
    return [str(item) for item in (ctx.get("features") or [])]


def _feature_info(ctx) -> pd.DataFrame:
    info = ctx.get("feature_info")
    if isinstance(info, pd.DataFrame):
        return info
    return pd.DataFrame(index=pd.Index(_feature_list(ctx), name="feature"))


def _column_values(info: pd.DataFrame, column: str, features: list[str]) -> dict[str, object]:
    if not isinstance(info, pd.DataFrame) or column not in info.columns:
        return {}
    out: dict[str, object] = {}
    series = info[column]
    index = {str(i): i for i in info.index}
    for feature in features:
        hit = index.get(feature)
        if hit is None:
            continue
        try:
            out[feature] = series.loc[hit]
        except Exception:  # noqa: BLE001
            continue
    return out


def _feature_mz(ctx) -> dict[str, float]:
    """特征 → m/z（优先 feature_info.mz，缺失再回退到特征 id 里的 mz 编码）。"""
    features = _feature_list(ctx)
    info = _feature_info(ctx)
    raw = _column_values(info, "mz", features)
    out: dict[str, float] = {}
    for feature in features:
        value = raw.get(feature)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        if not math.isfinite(number):
            match = re.search(r"mz[_\-\s]*([0-9]+(?:\.[0-9]+)?)", feature, re.IGNORECASE)
            if match:
                number = float(match.group(1))
        if math.isfinite(number) and number > 0:
            out[feature] = float(number)
    return out


def _feature_names(ctx) -> dict[str, str]:
    features = _feature_list(ctx)
    info = _feature_info(ctx)
    raw = _column_values(info, "name", features)
    out: dict[str, str] = {}
    for feature in features:
        value = raw.get(feature)
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in ("nan", "none", "na"):
            out[feature] = text
    return out


def _polarity(ctx) -> str:
    raw = str(ctx.get("polarity") or "").strip().lower()
    if raw.startswith("pos") or raw in ("+", "positive", "p"):
        return "pos"
    if raw.startswith("neg") or raw in ("-", "negative", "n"):
        return "neg"
    info = _feature_info(ctx)
    if isinstance(info, pd.DataFrame) and "polarity" in info.columns:
        values = [str(v).strip().lower() for v in info["polarity"].dropna().tolist()]
        n_pos = sum(1 for v in values if v.startswith("pos"))
        n_neg = sum(1 for v in values if v.startswith("neg"))
        if n_pos > n_neg:
            return "pos"
        if n_neg > n_pos:
            return "neg"
    return "unknown"


def _adduct_set(ctx, params) -> tuple[list[str], str]:
    """本次分析使用的加合物集合（全部来自 mb_common.ADDUCTS）与依据说明。"""
    chosen: list[str] = []
    info = _feature_info(ctx)
    for value in _column_values(info, "adduct", _feature_list(ctx)).values():
        text = str(value or "").strip()
        if text in ADDUCTS and text not in chosen:
            chosen.append(text)
    raw = params.get("adducts") or params.get("adduct")
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for value in raw:
            text = str(value or "").strip()
            if text in ADDUCTS and text not in chosen:
                chosen.append(text)
    if chosen:
        return chosen[:6], "特征注释/参数指定的加合物"
    polarity = _polarity(ctx)
    if polarity == "pos":
        return list(POSITIVE_ADDUCTS[:2]), "正离子：质子化 + 钠加合"
    if polarity == "neg":
        return list(NEGATIVE_ADDUCTS[:2]), "负离子：去质子 + 氯加合"
    return ["[M+H]+", "[M-H]-"], "极性未知：正、负离子主加合物都尝试"


def _feature_map(ctx, key: str) -> dict[str, set[str]]:
    """feature_info 的某列（分号/逗号分隔的多值）→ 特征 → 取值集合。"""
    features = _feature_list(ctx)
    info = _feature_info(ctx)
    raw = _column_values(info, key, features)
    out: dict[str, set[str]] = {}
    for feature, value in raw.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none", "na", "unknown"):
            continue
        parts = [p.strip() for p in _NAME_SPLIT.split(text) if p.strip()]
        if parts:
            out[feature] = set(parts)
    return out


def _resolve_label(name: str, library: dict[str, tuple[str, ...]]) -> str:
    """把注释里的通路名对到库里的规范名（完全一致 / 归一化一致 / 包含关系）。"""
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw in library:
        return raw
    low = raw.lower()
    for key in library:
        if key.lower() == low:
            return key
    target = _norm_name(raw)
    if target:
        for key in library:
            if _norm_name(key) == target:
                return key
    for key in library:
        if low in key.lower() or key.lower() in low:
            return key
    return raw


def _pathway_membership(ctx, library: dict[str, tuple[str, ...]], notes: list[str]):
    """特征 → 通路集合（注释优先，其次名称匹配，最后 KEGG 编号交集）。"""
    features = _feature_list(ctx)
    info = _feature_info(ctx)
    membership: dict[str, set[str]] = {f: set() for f in features}
    basis: list[str] = []

    # (1) feature_info.pathway / pathways / KEGG_PATHWAY 列：直接给通路名
    for column in ("pathway", "pathways", "kegg_pathway"):
        direct = _feature_map(ctx, column)
        if not direct:
            continue
        for feature, names in direct.items():
            for name in names:
                resolved = _resolve_label(name, library)
                if resolved:
                    membership[feature].add(resolved)
        basis.append(f"{column} 列（{len(direct)} 个特征有通路注释）")

    # (2) 名称匹配：特征 name → 通路成员名（归一化词干相等）
    names = _feature_names(ctx)
    if names:
        stem_index: dict[str, set[str]] = {}
        for path, members in library.items():
            for member in members:
                stem = _norm_name(member)
                if stem:
                    stem_index.setdefault(stem, set()).add(path)
        matched = 0
        for feature, text in names.items():
            stem = _norm_name(text)
            hit = stem_index.get(stem) if stem else None
            if hit:
                membership[feature] |= hit
                matched += 1
        if matched:
            basis.append(f"代谢物名匹配（{matched} 个特征命中内置通路库成员名）")

    # (3) KEGG 化合物编号匹配：只在通路库成员是 KEGG 编号（外部 genesets）时生效
    kegg = _feature_map(ctx, "kegg")
    if kegg:
        id_index: dict[str, set[str]] = {}
        for path, members in library.items():
            for member in members:
                if _KEGG_ID.match(str(member)):
                    id_index.setdefault(str(member), set()).add(path)
        matched = 0
        if id_index:
            for feature, ids in kegg.items():
                for ident in ids:
                    hit = id_index.get(ident)
                    if hit:
                        membership[feature] |= hit
                        matched += 1
                        break
        if matched:
            basis.append(f"KEGG 编号匹配（{matched} 个特征命中外部通路库的化合物编号）")

    # (4) KEGG 编号本身当通路名？没有意义；但 kegg 列若写的是通路名也接受
    if kegg:
        resolved_any = 0
        for feature, ids in kegg.items():
            for ident in ids:
                if _KEGG_ID.match(ident):
                    continue
                resolved = _resolve_label(ident, library)
                if resolved in library:
                    membership[feature].add(resolved)
                    resolved_any += 1
        if resolved_any:
            basis.append(f"kegg 列按通路名解析（{resolved_any} 条）")

    if basis:
        notes.append("通路归属依据：" + "；".join(dict.fromkeys(basis)))
    return membership


def _class_membership(ctx, notes: list[str]) -> tuple[dict[str, str], str]:
    """特征 → 化学分类（优先 feature_info.class，其次 HMDB 名称词典 hmdb_class_of）。"""
    features = _feature_list(ctx)
    info = _feature_info(ctx)
    classes = _feature_map(ctx, "class")
    if not classes:
        classes = _feature_map(ctx, "hmdb_class")
    basis = "feature_info.class"
    if not classes:
        names = _feature_names(ctx)
        classes = {}
        for feature, text in names.items():
            label = hmdb_class_of(text)
            if label and label != "Unknown":
                classes[feature] = {label}
        basis = "hmdb_class_of(name) 名称词典"
    if not classes:
        raise prerequisite(
            "特征表里没有化学分类注释（class / hmdb_class），也没有可识别的代谢物名，"
            "无法做 HMDB 化学分类富集；请导入带 class 列的特征表或先跑「代谢物鉴定」"
        )
    flat: dict[str, str] = {}
    for feature, labels in classes.items():
        ordered = sorted(labels)
        flat[feature] = ordered[0] if ordered else ""
    notes.append(f"化学分类依据：{basis}")
    return flat, basis


# ---------------------------------------------------------------------------
# 3. 差异表：取 ctx["results"]["de"] 并统一列名
# ---------------------------------------------------------------------------

DE_FEATURE_CANDIDATES = ("feature", "feature_id", "gene", "metabolite", "id", "name")
DE_FC_CANDIDATES = ("log2FC", "logFC", "log2_fc", "log2FoldChange", "log_fc", "FC")
DE_P_CANDIDATES = ("p", "P.Value", "pvalue", "p_value", "P")
DE_PADJ_CANDIDATES = ("p_adj", "adj.P.Val", "padj", "FDR", "fdr", "qvalue", "p.adjust", "P.adj")


def _de_frame(ctx) -> pd.DataFrame:
    """取差异分析结果（DataFrame 或 {"table": DataFrame}）并统一成 feature/log2FC/p/p_adj。"""
    results = ctx.get("results")
    if not isinstance(results, dict) or not results.get("de"):
        raise prerequisite("请先跑「单变量差异」（步骤 de）再做强")
    de = results.get("de")
    frame: pd.DataFrame | None = None
    if isinstance(de, pd.DataFrame):
        frame = de.copy()
    elif isinstance(de, dict):
        for key in ("table", "de_table", "tables", "rows", "de"):
            value = de.get(key)
            if isinstance(value, pd.DataFrame) and not value.empty:
                frame = value.copy()
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                frame = pd.DataFrame(value)
                break
    if frame is None or frame.empty:
        raise prerequisite("差异分析结果为空（results['de'] 里没有表），请重跑「单变量差异」")

    # 特征 id 列（只在没有 feature 列时才改名，避免与 name 列撞名产生重复列）
    if "feature" not in frame.columns:
        renamed = False
        for cand in DE_FEATURE_CANDIDATES:
            if cand in frame.columns:
                frame = frame.rename(columns={cand: "feature"})
                renamed = True
                break
        if not renamed:
            index_name = str(frame.index.name or "feature")
            frame = frame.reset_index()
            if index_name in frame.columns:
                frame = frame.rename(columns={index_name: "feature"})
            else:
                frame = frame.rename(columns={frame.columns[0]: "feature"})

    renames: dict[str, str] = {}
    for target, cands in (("log2FC", DE_FC_CANDIDATES), ("p", DE_P_CANDIDATES), ("p_adj", DE_PADJ_CANDIDATES)):
        if target in frame.columns:
            continue
        for cand in cands:
            if cand in frame.columns:
                renames[cand] = target
                break
    if renames:
        frame = frame.rename(columns=renames)

    frame = frame.copy()
    frame["feature"] = frame["feature"].astype(str)
    for column in ("log2FC", "p", "p_adj"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "log2FC" not in frame.columns:
        frame["log2FC"] = 0.0
    if "p" not in frame.columns and "p_adj" not in frame.columns:
        raise prerequisite("差异表里没有 p 值列（找不到 p / P.Value / pvalue），无法做富集")
    if "p" not in frame.columns:
        frame["p"] = frame["p_adj"]
    if "p_adj" not in frame.columns:
        frame["p_adj"] = bh_fdr(frame["p"].to_numpy(float))
    frame["p"] = frame["p"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    frame["p_adj"] = frame["p_adj"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    frame["log2FC"] = frame["log2FC"].fillna(0.0)
    frame = frame[frame["feature"].astype(str).str.len() > 0]
    return frame.drop_duplicates("feature").reset_index(drop=True)


def _significant(de: pd.DataFrame, p_cut: float, fc: float | None = None) -> pd.DataFrame:
    """显著特征：p_adj < p_cut（可选再要求 |FC| ≥ fc），按 |log2FC| 降序。

    ``fc`` 是**倍数**（如 1.5 = 50% 变化），log2FC 是 log2 尺度，两者必须先换算到同一口径：
    |log2FC| ≥ log2(fc)（BIO_P0_BACKLOG §4 P0-3 的 fc/log2FC 口径一致性）。旧实现直接拿
    |log2FC| 和倍数比较，fc=1.5 实际卡在 2^1.5≈2.83 倍，等于默认更严。
    """
    mask = de["p_adj"].to_numpy(float) < float(p_cut)
    if fc is not None:
        try:
            fold = float(fc)
            if np.isfinite(fold) and fold > 1.0:
                mask &= np.abs(de["log2FC"].to_numpy(float)) >= math.log2(fold) - 1e-12
        except (TypeError, ValueError):
            pass
    sub = de.loc[mask].copy()
    sub["_abs"] = np.abs(sub["log2FC"].to_numpy(float))
    sub = sub.sort_values(["_abs", "p_adj"], ascending=[False, True]).drop(columns=["_abs"])
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. 内置库 / 外部 genesets / 可选在线补充
# ---------------------------------------------------------------------------


def _as_members(value) -> tuple[list[str], list[float]]:
    """把一条通路定义解析成（成员名列表, 额外中性质量列表）。"""
    names: list[str] = []
    masses: list[float] = []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        for key in ("metabolites", "members", "compounds", "genes", "names", "metabolite"):
            if key in value:
                return _as_members(value.get(key))
        # {"name": "...", "mass": 123.4} 这种直接给质量的写法
        if "mass" in value:
            try:
                masses.append(float(value["mass"]))
            except (TypeError, ValueError):
                pass
        for key in ("name", "id"):
            if value.get(key):
                names.append(str(value[key]))
        return names, masses
    if isinstance(value, (list, tuple, set)):
        for item in value:
            sub_names, sub_masses = _as_members(item)
            names.extend(sub_names)
            masses.extend(sub_masses)
    elif value is not None:
        names.append(str(value))
    return names, masses


def _read_genesets(workdir: str, notes: list[str], warnings: list[str]):
    """读取 <workdir>/genesets/*.json 用户/在线缓存；格式兜底见 _as_members。"""
    library: dict[str, list[str]] = {}
    extra_masses: dict[str, list[float]] = {}
    if not workdir or not os.path.isdir(workdir):
        return library, extra_masses
    patterns = [os.path.join(workdir, "genesets", "*.json"), os.path.join(workdir, "genesets", "*.jsonl")]
    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        return library, extra_masses
    used = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            warnings.append(f"读取通路缓存 {os.path.basename(path)} 失败：{error}")
            continue
        entries: list[tuple[str, object]] = []
        try:
            if path.endswith(".jsonl"):
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        name = item.get("pathway") or item.get("name") or item.get("id") or item.get("term")
                        if name:
                            entries.append((str(name), item))
            else:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    inner = payload.get("pathways") if isinstance(payload.get("pathways"), (dict, list)) else None
                    if inner is None and isinstance(payload.get("data"), (dict, list)):
                        inner = payload.get("data")
                    if isinstance(inner, (dict, list)):
                        payload = inner
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            if str(key).lower() in ("meta", "version", "source", "url", "date"):
                                continue
                            entries.append((str(key), value))
                    else:
                        for item in payload:
                            if not isinstance(item, dict):
                                continue
                            name = item.get("pathway") or item.get("name") or item.get("id") or item.get("term")
                            if name:
                                entries.append((str(name), item))
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            name = item.get("pathway") or item.get("name") or item.get("id") or item.get("term")
                            if name:
                                entries.append((str(name), item))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            warnings.append(f"通路缓存 {os.path.basename(path)} 不是合法 JSON（{error}），已跳过")
            continue
        for name, value in entries:
            names, masses = _as_members(value)
            names = [n for n in dict.fromkeys(names) if n and len(n) < 200]
            if not names:
                continue
            bucket = library.setdefault(name, [])
            for member in names:
                if member not in bucket:
                    bucket.append(member)
            if masses:
                extra_masses.setdefault(name, []).extend(masses)
            used += 1
    if used:
        notes.append(f"已加载本地通路缓存 {len(files)} 个文件、{len(library)} 条通路（{workdir}/genesets）")
    return library, extra_masses


def _parse_kegg_pathway(text: str) -> list[str]:
    """解析 rest.kegg.jp/get/<pathway> 的 COMPOUND 段为代谢物名列表。"""
    names: list[str] = []
    in_compound = False
    for line in text.splitlines():
        if line and not line.startswith(" "):
            in_compound = line.strip().upper() == "COMPOUND"
            continue
        if not in_compound:
            continue
        body = line.strip()
        if not body:
            continue
        token, _, rest = body.partition("  ")
        rest = rest.strip() or token
        label = rest.split(";")[0].strip()
        if label and len(label) < 200:
            names.append(label)
    return names


def _online_library(workdir: str, library: dict[str, tuple[str, ...]],
                    notes: list[str], warnings: list[str], timeout: float = 5.0):
    """可选在线补充（rest.kegg.jp，DESIGN §10 说明可达）；失败只 note/warning，不阻塞。"""
    cache_dir = os.path.join(workdir, "genesets") if workdir else ""
    cache_path = os.path.join(cache_dir, "kegg_online.json") if cache_dir else ""
    if cache_path and os.path.isfile(cache_path):
        cached, extra = _read_genesets(workdir, [], [])
        if cached:
            notes.append(f"复用在线通路缓存 {cache_path}")
            merged = dict(library)
            for name, members in cached.items():
                merged[name] = tuple(dict.fromkeys(list(merged.get(name, ())) + list(members)))
            return merged, extra
    merged = dict(library)
    fetched = 0
    failures = 0
    for path in list(library.keys())[:40]:
        match = re.search(r"hsa\d{5}|map\d{5}|ko\d{5}", path)
        target = match.group(0) if match else None
        if not target:
            continue
        url = f"https://rest.kegg.jp/get/{target}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - 固定 https 端点
                text = response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - 网络失败按降级处理
            failures += 1
            warnings.append(f"在线补充 KEGG {target} 失败（{type(error).__name__}），沿用内置库")
            continue
        members = _parse_kegg_pathway(text)
        if members:
            merged[path] = tuple(dict.fromkeys(list(merged.get(path, ())) + members))
            fetched += 1
    if fetched:
        notes.append(f"在线补充 {fetched} 条 KEGG 通路成员（rest.kegg.jp）")
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                payload = {name: list(members) for name, members in merged.items()}
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
            except OSError:
                pass
    elif failures:
        notes.append("rest.kegg.jp 不可达或未命中，已使用内置通路库（含本地 genesets 缓存）")
    return merged, {}


def _load_library(ctx, params, notes: list[str], warnings: list[str]):
    """合并内置库 + 本地 genesets 缓存 +（可选）在线补充。"""
    workdir = str(ctx.get("workdir") or "")
    library: dict[str, tuple[str, ...]] = {name: tuple(members) for name, members in MB_PATHWAY_MEMBERS.items()}
    local, local_masses = _read_genesets(workdir, notes, warnings)
    for name, members in local.items():
        library[name] = tuple(dict.fromkeys(list(library.get(name, ())) + list(members)))
    extra_masses = {name: list(values) for name, values in local_masses.items()}
    online = str(params.get("online") or "").strip().lower() in ("1", "true", "yes", "on")
    if online:
        library, online_masses = _online_library(workdir, library, notes, warnings)
        for name, values in (online_masses or {}).items():
            extra_masses.setdefault(name, []).extend(values)
    if len(library) < 5:
        warnings.append(f"通路库只剩 {len(library)} 条，富集结果可能不可靠")
    return library, extra_masses


# ---------------------------------------------------------------------------
# 5. 质量匹配（mummichog 的核心）
# ---------------------------------------------------------------------------


def _theoretical_mz(neutral_masses: list[float], adducts: list[str]) -> np.ndarray:
    """中性质量 → 理论离子 m/z 数组（用 mb_common.ADDUCTS 的加合物位移/电荷）。"""
    out: list[float] = []
    for neutral in neutral_masses:
        base = float(neutral)
        if not math.isfinite(base) or base <= 0:
            continue
        for adduct in adducts:
            shift, charge = ADDUCTS.get(adduct, (0.0, 1))
            charge = int(charge) if int(charge) else 1
            value = (base + float(shift)) / charge
            if math.isfinite(value) and value > 0:
                out.append(float(value))
    if not out:
        return np.zeros(0, dtype=float)
    return np.unique(np.round(np.asarray(out, dtype=float), 6))


def _pathway_mz_table(library, extra_masses, adducts, min_size: int):
    """每条通路 → 理论离子 m/z 数组（通路成员太少时跳过）。"""
    theo: dict[str, np.ndarray] = {}
    for path, members in library.items():
        neutrals: list[float] = []
        for member in members:
            mass = MB_METABOLITE_MASS.get(member)
            if mass is not None:
                neutrals.append(float(mass))
        neutrals.extend(float(v) for v in (extra_masses or {}).get(path, []) if math.isfinite(float(v)))
        if len(members) < max(1, int(min_size)):
            continue
        values = _theoretical_mz(neutrals, adducts)
        if values.size:
            theo[path] = values
    return theo


def _mz_hit_mask(mz_values: np.ndarray, theoretical: np.ndarray, tol_ppm: float) -> np.ndarray:
    """观测 m/z 是否落在理论离子质量的 ppm 容差内（最近邻 searchsorted）。"""
    mask = np.zeros(mz_values.size, dtype=bool)
    if theoretical.size == 0 or mz_values.size == 0:
        return mask
    ok = np.isfinite(mz_values) & (mz_values > 0)
    if not ok.any():
        return mask
    observed = mz_values[ok]
    ordered = np.sort(np.asarray(theoretical, dtype=float))
    idx = np.searchsorted(ordered, observed)
    left = ordered[np.clip(idx - 1, 0, ordered.size - 1)]
    right = ordered[np.clip(idx, 0, ordered.size - 1)]
    err = np.minimum(np.abs(observed - left), np.abs(observed - right)) / np.maximum(observed, 1e-9) * 1e6
    mask[np.where(ok)[0]] = err <= float(tol_ppm)
    return mask


def _hit_labels(ctx, features: list[str], limit: int = MAX_HIT_LABELS) -> str:
    names = _feature_names(ctx)
    parts: list[str] = []
    for feature in features[:limit]:
        label = names.get(feature)
        parts.append(f"{label} ({feature})" if label else feature)
    if len(features) > limit:
        parts.append(f"… 共 {len(features)} 个")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# 6. 富集引擎
# ---------------------------------------------------------------------------


def _mummichog_with_fallback(ctx, sig: pd.DataFrame, params: dict, library, extra_masses,
                              notes: list[str], warnings: list[str]):
    """跑 Mummichog；无显著通路但注释 ORA 有显著结果时换用 ORA（写 warnings 说明方法变更）。

    返回 (frame, method_text, label_key)。DEMO/随机质量数据上 m/z 匹配不到真实代谢物，
    而特征表又带 pathway 注释时，这是唯一能给出生物学解释的路径（Lead 明确要求优先消费注释）。
    """
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig_features = sig["feature"].astype(str).tolist()
    frame, status = _mummichog(ctx, sig, params, library, extra_masses, notes, warnings)
    base_method = "内置实现（Mummichog 质量映射 + 置换；依据 Li 2013 / Uppal 2017）"
    if not frame.empty and int((frame["p_adj"] < p_cut).sum()) > 0:
        return frame, base_method, "pathway"

    reason = {
        "mz_missing": "特征表没有 m/z，无法做质量匹配",
        "no_mass_hit": f"{len(sig_features)} 个显著特征的 m/z 未命中任何通路的理论离子质量",
        "library_empty": "通路库为空",
        "sig_not_in_matrix": "显著特征不在当前矩阵里（可能差异分析后矩阵变过）",
    }.get(status, "") if frame.empty else (
        f"质量匹配命中 {int(frame['n_hits'].sum())} 个特征，但没有通路达到 FDR<{p_cut:g}"
    )
    notes.append(f"{reason}；尝试改用注释 ORA")
    membership = _pathway_membership(ctx, library, notes)
    if any(membership.get(f) for f in sig_features):
        ora_frame = _ora(ctx, membership, "pathway", sig_features, params, notes)
        if not ora_frame.empty and int((ora_frame["p_adj"] < p_cut).sum()) > 0:
            warnings.append(f"{reason}，已改用注释 ORA（结果解读请注明方法变更）")
            return ora_frame, "内置实现（质量匹配不显著 → KEGG 通路 ORA 超几何 + BH；依据 Kanehisa 2023）", "pathway"
    if not frame.empty:
        warnings.append(f"{reason}；注释 ORA 也没有显著通路，保留质量匹配结果供参考")
        return frame, base_method, "pathway"
    raise prerequisite(
        f"{reason}；特征表也没有可用的通路注释（pathway/kegg/代谢物名），无法回退 ORA。"
        "请导入带 mz 与通路注释的特征表，或先补齐注释",
    )


def _enrich_frame(rows: list[dict], label_key: str = "pathway") -> pd.DataFrame:
    columns = [label_key, "n_hits", "n_total", "ratio", "fold", "p", "p_adj", "hits"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    for column in ("n_hits", "n_total", "ratio", "fold", "p", "p_adj"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["p", "p_adj", "n_hits"], ascending=[True, True, False]).reset_index(drop=True)


def _mummichog(ctx, sig: pd.DataFrame, params: dict, library, extra_masses,
               notes: list[str], warnings: list[str]):
    """Mummichog 式质量映射 + 置换（Li 2013；Uppal 2017）。"""
    mz_map = _feature_mz(ctx)
    if not mz_map:
        return pd.DataFrame(), "mz_missing"
    ppm = float(params.get("ppm") or DEFAULT_PPM)
    n_perm = int(params.get("n_perm") or DEFAULT_N_PERM)
    min_size = int(params.get("min_size") or DEFAULT_MIN_SIZE)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    adducts, adduct_note = _adduct_set(ctx, params)
    theo = _pathway_mz_table(library, extra_masses, adducts, min_size)
    if not theo:
        return pd.DataFrame(), "library_empty"

    features = _feature_list(ctx)
    mz_values = np.asarray([mz_map.get(f, float("nan")) for f in features], dtype=float)
    paths = list(theo.keys())
    masks = np.zeros((len(features), len(paths)), dtype=bool)
    for j, path in enumerate(paths):
        masks[:, j] = _mz_hit_mask(mz_values, theo[path], ppm)

    sig_features = [f for f in sig["feature"].astype(str).tolist() if f in set(features)]
    if not sig_features:
        return pd.DataFrame(), "sig_not_in_matrix"
    sig_index = np.asarray([features.index(f) for f in sig_features], dtype=int)
    pool = np.where(np.isfinite(mz_values))[0]
    if pool.size == 0:
        return pd.DataFrame(), "mz_missing"
    observed = masks[sig_index].sum(axis=0)
    if int(observed.sum()) == 0:
        return pd.DataFrame(), "no_mass_hit"

    rng = np.random.default_rng(SEED)
    n_draw = int(min(sig_index.size, pool.size))
    counts = np.zeros((max(1, n_perm), len(paths)), dtype=np.int32)
    for b in range(max(1, n_perm)):
        sample = rng.choice(pool, size=n_draw, replace=False)
        counts[b] = masks[sample].sum(axis=0)
    p_values = (1.0 + (counts >= observed[None, :]).sum(axis=0)) / (1.0 + max(1, n_perm))
    q_values = bh_fdr(p_values)

    rows: list[dict] = []
    for j, path in enumerate(paths):
        n_hits = int(observed[j])
        if n_hits <= 0:
            continue
        members = list(library.get(path, ()))
        n_total = len(members)
        hit_names = _hit_labels(ctx, [f for f in sig_features if masks[features.index(f), j]])
        expected = max(1e-9, n_draw * n_total / max(len(features), 1))
        rows.append({
            "pathway": path,
            "n_hits": n_hits,
            "n_total": n_total,
            "ratio": float(n_hits) / max(n_total, 1),
            "fold": float(n_hits) / expected,
            "p": float(p_values[j]),
            "p_adj": float(q_values[j]),
            "hits": hit_names,
        })
    frame = _enrich_frame(rows)
    notes.append(
        f"Mummichog 质量匹配：{len(sig_features)} 个显著特征、{len(theo)} 条通路、"
        f"{ppm:g} ppm、加合物 {'/'.join(adducts)}（{adduct_note}）；置换 {n_perm} 次（seed={SEED}）"
    )
    if not frame.empty and (frame["p"] < p_cut).sum() == 0:
        warnings.append(f"质量匹配到 {int(frame['n_hits'].sum())} 个命中，但没有通路达到 p<{p_cut:g}")
    return frame, "ok"


def _ora(ctx, membership: dict[str, set[str]], label_key: str, sig_features: list[str],
         params: dict, notes: list[str], min_size: int | None = None) -> pd.DataFrame:
    """超几何（单尾 Fisher）+ BH；背景 = 有该类注释的全部特征。"""
    min_size = int(min_size if min_size is not None else (params.get("min_size") or DEFAULT_MIN_SIZE))
    features = _feature_list(ctx)
    background = {f for f in features if membership.get(f)}
    if not background:
        raise prerequisite("特征表里没有可用于富集的通路/分类注释（注释为空），请先补充注释")
    sig = {f for f in sig_features if f in background}
    if not sig:
        raise prerequisite("显著差异特征里没有带通路注释的特征，无法做 ORA（可改用 Mummichog 质量匹配）")
    labels: dict[str, set[str]] = {}
    for feature in background:
        for label in membership.get(feature, ()):  # 一个特征可属多条通路
            if not label:
                continue
            labels.setdefault(str(label), set()).add(feature)
    n_background = len(background)
    n_sig = len(sig)
    try:
        from scipy import stats as _stats  # noqa: PLC0415 - scipy 已随环境安装

        sf = _stats.hypergeom.sf
    except Exception:  # noqa: BLE001 - 理论上不会发生
        sf = None

    rows: list[dict] = []
    for label, members in labels.items():
        hits = members & sig
        if not hits:
            continue
        n_total = len(members)
        if n_total < min_size:
            continue
        n_hits = len(hits)
        if sf is not None:
            p_value = float(sf(n_hits - 1, n_background, n_total, n_sig))
        else:  # 兜底：不带 scipy 时用正态近似（极少用）
            expected = n_sig * n_total / max(n_background, 1)
            p_value = 1.0 if n_hits <= expected else math.erfc(
                (n_hits - expected) / math.sqrt(max(expected * (1 - n_total / max(n_background, 1)), 1e-9))
            )
        expected = n_sig * n_total / max(n_background, 1)
        rows.append({
            label_key: label,
            "n_hits": n_hits,
            "n_total": n_total,
            "ratio": float(n_hits) / max(n_total, 1),
            "fold": float(n_hits) / max(expected, 1e-9),
            "p": min(max(p_value, 1e-300), 1.0),
            "p_adj": float("nan"),
            "hits": _hit_labels(ctx, sorted(hits)),
        })
    if not rows:
        return _enrich_frame([], label_key)
    frame = _enrich_frame(rows, label_key)
    frame["p_adj"] = bh_fdr(frame["p"].to_numpy(float))
    frame = frame.sort_values(["p", "p_adj", "n_hits"], ascending=[True, True, False]).reset_index(drop=True)
    notes.append(
        f"ORA 超几何：{n_sig} 个显著特征 / {n_background} 个有注释特征、"
        f"{len(frame)} 条通路（min_size={min_size}），BH 校正"
    )
    return frame


def _msea(ctx, de: pd.DataFrame, annotation: dict[str, set[str]], label_key: str,
          params: dict, notes: list[str]) -> pd.DataFrame:
    """MSEA：按 log2FC 排序的加权 KS 游走 + 置换（Subramanian 2005；Xia 2015）。"""
    n_perm = int(params.get("n_perm") or DEFAULT_N_PERM)
    min_size = int(params.get("min_size") or 5)
    features = _feature_list(ctx)
    keep = set(features)
    frame = de[de["feature"].astype(str).isin(keep)].copy()
    if frame.empty:
        raise prerequisite("差异表与矩阵的特征对不上（没有交集），请重跑差异分析")
    frame = frame.sort_values("log2FC", ascending=False).reset_index(drop=True)
    ranked = frame["feature"].astype(str).tolist()
    weights = np.abs(frame["log2FC"].to_numpy(float))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    position = {feature: i for i, feature in enumerate(ranked)}
    n = len(ranked)

    sets: dict[str, list[int]] = {}
    for feature, labels in annotation.items():
        idx = position.get(feature)
        if idx is None:
            continue
        for label in labels:
            sets.setdefault(str(label), []).append(idx)
    sets = {label: sorted(v) for label, v in sets.items()}
    if not sets:
        raise prerequisite("差异特征没有通路/分类注释，无法做代谢物集富集（MSEA）")

    def es_of(indices: np.ndarray, w: np.ndarray) -> float:
        k = indices.size
        if k == 0 or k >= n:
            return 0.0
        marked = np.zeros(n, dtype=float)
        marked[indices] = 1.0
        hit_w = w[indices]
        total = float(hit_w.sum())
        if total <= 0:
            hit_w = np.ones(k, dtype=float)
            total = float(k)
        wsum = np.zeros(n, dtype=float)
        wsum[indices] = hit_w
        running = np.cumsum(wsum) / total - np.cumsum(1.0 - marked) / float(n - k)
        return float(running.max())

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for label, indices in sets.items():
        k = len(indices)
        if k < min_size:
            continue
        observed = es_of(np.asarray(indices, dtype=int), weights)
        null = np.zeros(max(1, n_perm), dtype=float)
        for b in range(max(1, n_perm)):
            null[b] = es_of(rng.choice(n, size=k, replace=False), weights)
        p_value = (1.0 + float((null >= observed).sum())) / (1.0 + max(1, n_perm))
        scale = float(np.mean(np.abs(null))) if null.size else 0.0
        rows.append({
            label_key: label,
            "n_hits": k,
            "n_total": k,
            "ratio": float(k) / max(n, 1),
            "fold": float("nan"),
            "p": min(max(p_value, 1e-300), 1.0),
            "p_adj": float("nan"),
            "hits": _hit_labels(ctx, [ranked[i] for i in indices[:MAX_HIT_LABELS]]),
            "es": observed,
            "nes": observed / scale if scale > 0 else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    frame_out = pd.DataFrame(rows)
    frame_out["p_adj"] = bh_fdr(frame_out["p"].to_numpy(float))
    frame_out = frame_out.sort_values(["p", "p_adj", "es"], ascending=[True, True, False]).reset_index(drop=True)
    notes.append(
        f"MSEA：{n} 个特征按 log2FC 排序、{len(frame_out)} 条集合（min_size={min_size}）、"
        f"置换 {n_perm} 次（seed={SEED}，带符号加权 KS）"
    )
    return frame_out


# ---------------------------------------------------------------------------
# 7. 图
# ---------------------------------------------------------------------------


def _figure_bar(frame: pd.DataFrame, outdir: str, name: str, title: str,
                top_n: int = 15, p_cut: float = DEFAULT_P_CUT, label_key: str = "pathway") -> dict:
    sub = frame.copy()
    sub = sub.sort_values("p", ascending=True).head(max(1, int(top_n))).iloc[::-1]
    p_values = np.clip(sub["p"].to_numpy(float), 1e-300, 1.0)
    q_values = np.clip(sub["p_adj"].to_numpy(float), 1e-300, 1.0) if "p_adj" in sub.columns else p_values
    values = -np.log10(p_values)
    labels = [str(v) for v in sub[label_key].tolist()]
    height = max(3.4, 0.38 * len(labels) + 1.5)
    fig, ax = new_fig(10.2, height)
    colors = [PALETTE[0] if q < float(p_cut) else "#b9bfc8" for q in q_values]
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors, height=0.68, edgecolor="white", linewidth=0.4)
    ax.set_yticks(positions)
    short = [label if len(label) <= 44 else label[:42] + "…" for label in labels]
    ax.set_yticklabels(short, fontsize=8.4)
    ax.set_xlabel("-log10(P)")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.axvline(-math.log10(max(p_cut, 1e-300)), color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.8)
    top = float(values.max()) if values.size else 1.0
    for pos, value, q in zip(positions, values, q_values):
        ax.text(value + max(top, 1.0) * 0.015, pos, f"FDR {q:.2g}", va="center", fontsize=7.2, color="#5a6068")
    ax.set_xlim(0, max(top * 1.22, -math.log10(max(p_cut, 1e-300)) * 1.4))
    ax.grid(axis="x", color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    return save_fig(fig, outdir, name, title)


def _figure_dot(frame: pd.DataFrame, outdir: str, name: str, title: str,
                top_n: int = 15, label_key: str = "pathway") -> dict:
    sub = frame.copy().sort_values("p", ascending=True).head(max(1, int(top_n))).iloc[::-1]
    labels = [str(v) for v in sub[label_key].tolist()]
    ratio = sub["ratio"].to_numpy(float) if "ratio" in sub.columns else np.zeros(len(sub))
    hits = sub["n_hits"].to_numpy(float) if "n_hits" in sub.columns else np.ones(len(sub))
    q_values = np.clip(sub["p_adj"].to_numpy(float), 1e-300, 1.0) if "p_adj" in sub.columns else np.ones(len(sub))
    height = max(3.4, 0.38 * len(labels) + 1.8)
    fig, ax = new_fig(9.6, height)
    positions = np.arange(len(labels))
    sizes = 26.0 + 13.0 * hits
    scatter = ax.scatter(ratio, positions, s=sizes, c=-np.log10(q_values), cmap="viridis",
                         edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([label if len(label) <= 44 else label[:42] + "…" for label in labels], fontsize=8.4)
    ax.set_xlabel("Hit ratio (hits / pathway size)")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="x", color="#e8eaee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0.0)
    for pos, hit in zip(positions, hits):
        ax.text(ax.get_xlim()[1] * 0.985, pos, f"n={int(hit)}", va="center", ha="right",
                fontsize=7.2, color="#8a9099")
    try:
        bar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
        bar.set_label("-log10(FDR)", fontsize=8.5)
        bar.ax.tick_params(labelsize=7.5)
    except Exception:  # noqa: BLE001 - colorbar 失败不影响图本身
        pass
    return save_fig(fig, outdir, name, title)


# ---------------------------------------------------------------------------
# 8. 结果信封
# ---------------------------------------------------------------------------


def _table_columns(label_key: str, label_title: str, extra: list[dict] | None = None) -> list[dict]:
    columns = [
        {"key": label_key, "label": label_title, "type": "string"},
        {"key": "n_hits", "label": "命中特征数", "type": "number"},
        {"key": "n_total", "label": "集合大小", "type": "number"},
        {"key": "ratio", "label": "命中比例", "type": "number"},
        {"key": "fold", "label": "富集倍数", "type": "number"},
        {"key": "p", "label": "P 值", "type": "number"},
        {"key": "p_adj", "label": "FDR", "type": "number"},
        {"key": "hits", "label": "命中特征", "type": "string"},
    ]
    if extra:
        columns.extend(extra)
    return columns


def _store_frame(frame: pd.DataFrame) -> dict:
    return {"table": frame, "method": "内置实现", "n_rows": int(len(frame))}


def _enrich_payload(frame: pd.DataFrame, label_key: str, label_title: str, title: str,
                    method_text: str, notes: list[str], warnings: list[str], outdir: str,
                    params: dict, metrics: dict | None = None, store_extra: dict | None = None,
                    file_stem: str = "enrich", fig_title: str = "Pathway enrichment") -> dict:
    """富集分析/步骤的统一返回：表 + 两张图 + data_out + store。"""
    top_n = int(params.get("top_n") or 15)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    figures: list[dict] = []
    tables: list[dict] = []
    data_out: list[dict] = []
    if not frame.empty:
        columns = _table_columns(label_key, label_title)
        if label_key != "pathway":
            pass
        tables.append(table("enrich", f"{label_title}富集结果", columns, frame.to_dict(orient="records"),
                            note=method_text))
        figures.append(_figure_bar(frame, outdir, f"{file_stem}-bar", f"{fig_title}: -log10(P)",
                                   top_n=top_n, p_cut=p_cut, label_key=label_key))
        figures.append(_figure_dot(frame, outdir, f"{file_stem}-dot", f"{fig_title}: hit ratio",
                                   top_n=top_n, label_key=label_key))
        try:
            path = export_csv(frame, outdir, f"{file_stem}.csv", index=False)
            data_out.append(data_file("enrich_csv", path, "通路富集表（完整）"))
        except Exception as error:  # noqa: BLE001 - 写文件失败只降级
            warnings.append(f"导出富集表失败：{error}")
    else:
        # 零显著/零命中是合法结果：仍然返回带列名的空表（total=0），不伪造行、不画无数据的图
        tables.append(table(
            "enrich", f"{label_title}富集结果（空）", _table_columns(label_key, label_title), rows=[],
            note=method_text + "；本次没有可报告的富集条目（空结果，不是失败）"))
    bullets = _bullets(frame, label_key, method_text, params, label_title)
    summary_metrics = {
        "n_features": int(metrics.get("n_features", 0)) if metrics else 0,
        "n_significant": int(metrics.get("n_significant", 0)) if metrics else 0,
        "n_pathways": int(len(frame)),
        "n_sig_pathways": int((frame["p_adj"] < p_cut).sum()) if not frame.empty else 0,
        "n_hits": int(frame["n_hits"].sum()) if not frame.empty else 0,
        "method": method_text,
    }
    if metrics:
        for key, value in metrics.items():
            summary_metrics.setdefault(key, value)
    store = {"enrich": {**(store_extra or {}), "table": frame, "method": method_text,
                        "label_key": label_key, "params": dict(params), "metrics": summary_metrics}}
    return result(title=title, bullets=bullets, metrics=summary_metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


def _bullets(frame: pd.DataFrame, label_key: str, method_text: str, params: dict,
             label_title: str = "通路") -> list[str]:
    bullets: list[str] = [method_text]
    if frame.empty:
        bullets.append("没有可报告的富集条目（命中为 0）")
        return bullets
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig = frame[frame["p_adj"] < p_cut]
    top = frame.iloc[0]
    bullets.append(
        f"共 {len(frame)} 条{label_title}有命中；{len(sig)} 条 FDR<{p_cut:g}；"
        f"最强信号：{top[label_key]}（{int(top['n_hits'])}/{int(top['n_total'])}，FDR={float(top['p_adj']):.3g}）"
    )
    if len(sig):
        names = "、".join(str(v) for v in sig.head(5)[label_key].tolist())
        bullets.append(f"显著{label_title}：{names}")
    bullets.append("方法：内置实现（质量匹配/超几何 + BH FDR；置换用固定 seed，可复现）")
    return bullets


# ---------------------------------------------------------------------------
# 9. 步骤：enrich
# ---------------------------------------------------------------------------


def _zero_significant_payload(ctx, params: dict, outdir: str, *, title: str,
                              method_text: str, label_key: str = "pathway",
                              label_title: str = "通路", fig_title: str = "Pathway enrichment",
                              p_cut: float = DEFAULT_P_CUT, file_stem: str = "enrich") -> dict:
    """零显著是合法结果：返回带列名的空表 + n_significant=0 + warning（不退化为 top50）。"""
    notes = ["零显著时不做全量排序富集；如需基于全基因/全代谢物排序的通路富集，"
             "请单独运行 metabolite_set_enrich（MSEA，秩基 GSEA 思想）"]
    warnings = [f"p_adj<{p_cut:g} 的显著特征为 0（零显著是合法结果，不是富集失败）："
                f"富集结果为空表"]
    empty = _enrich_frame([], label_key)
    return _enrich_payload(empty, label_key, label_title, title, method_text, notes, warnings,
                           outdir, params,
                           metrics={"n_features": int(len(_feature_list(ctx))), "n_significant": 0},
                           store_extra={"significant": []}, fig_title=fig_title,
                           file_stem=file_stem)


def step_enrich(ctx, params, outdir):
    """STEP enrich：默认 Mummichog 质量匹配；method=ora_kegg 时走注释超几何。

    零显著（p_adj < p_cut 的特征为 0）是**合法结果**：返回带列名的空表、n_significant=0
    与 warning，绝不退化成「按 |log2FC| 取 top50 再计为显著」（BIO_P0_BACKLOG §4 P0-3）。
    全量排序富集由 ``metabolite_set_enrich``（MSEA）单独承担。
    """
    params = dict(params or {})
    method = str(params.get("method") or "mummichog").strip().lower()
    notes: list[str] = []
    warnings: list[str] = []
    de = _de_frame(ctx)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    if method in ("ora_kegg", "ora", "kegg"):
        title = "KEGG 通路 ORA"
        method_text = "内置实现（KEGG 通路 ORA 超几何 + BH；依据 Kanehisa 2023）"
        fig_title = "KEGG pathway ORA"
    elif method in ("mummichog", "mummichog_pathway", "auto"):
        title = "Mummichog 式通路富集"
        method_text = "内置实现（Mummichog 质量映射 + 置换；依据 Li 2013 / Uppal 2017）"
        fig_title = "Mummichog pathway enrichment"
    else:
        raise bad_request(f"未知富集方法「{method}」（可选：mummichog / ora_kegg）")

    sig = _significant(de, p_cut, params.get("fc"))
    n_features = int(len(_feature_list(ctx)))
    if sig.empty:
        reason = f"p_adj<{p_cut:g} 的显著特征为 0（零显著是合法结果，不是富集失败）"
        warnings.append(reason + "：富集结果为空表；不再退化为按 |log2FC| 取 top50 当显著特征")
        notes.append(
            "零显著时不做全量排序富集；如需基于全基因/全代谢物排序的通路富集，"
            "请单独运行 metabolite_set_enrich（MSEA，秩基 GSEA 思想）"
        )
        empty = _enrich_frame([], "pathway")
        return _enrich_payload(empty, "pathway", "通路", title, method_text, notes, warnings,
                               outdir, params,
                               metrics={"n_features": n_features, "n_significant": 0},
                               store_extra={"significant": []},
                               fig_title=fig_title)
    if len(sig) > MAX_SIG:
        warnings.append(f"显著特征 {len(sig)} 个超过上限，按 |log2FC| 取前 {MAX_SIG} 个")
        sig = sig.head(MAX_SIG)
    library, extra_masses = _load_library(ctx, params, notes, warnings)
    sig_features = [f for f in sig["feature"].astype(str).tolist()]
    metrics = {"n_features": n_features, "n_significant": int(len(sig_features))}

    if method in ("ora_kegg", "ora", "kegg"):
        membership = _pathway_membership(ctx, library, notes)
        frame = _ora(ctx, membership, "pathway", sig_features, params, notes)
        label_key, label_title = "pathway", "通路"
    else:
        frame, method_text, label_key = _mummichog_with_fallback(
            ctx, sig, params, library, extra_masses, notes, warnings)
        label_title = "通路"

    return _enrich_payload(frame, label_key, label_title, title, method_text, notes, warnings,
                           outdir, params, metrics=metrics,
                           store_extra={"significant": sig_features[:500]},
                           fig_title=fig_title)


# ---------------------------------------------------------------------------
# 10. 分析（group=通路与化学分类）
# ---------------------------------------------------------------------------


def analysis_mummichog_pathway(ctx, params, outdir):
    """Mummichog 式通路富集（Li 2013）：质量匹配 + 置换经验 p，BH 校正。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    de = _de_frame(ctx)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig = _significant(de, p_cut, params.get("fc"))
    if sig.empty:
        return _zero_significant_payload(
            ctx, params, outdir, title="Mummichog 式通路富集",
            method_text="内置实现（Mummichog 质量映射 + 置换；依据 Li 2013 / Uppal 2017）",
            fig_title="Mummichog pathway enrichment", p_cut=p_cut)
    if len(sig) > MAX_SIG:
        sig = sig.head(MAX_SIG)
        warnings.append(f"显著特征超过上限，按 |log2FC| 取前 {MAX_SIG} 个")
    library, extra_masses = _load_library(ctx, params, notes, warnings)
    sig_features = sig["feature"].astype(str).tolist()
    frame, method_text, _label_key = _mummichog_with_fallback(
        ctx, sig, params, library, extra_masses, notes, warnings)
    return _enrich_payload(frame, "pathway", "通路", "Mummichog 式通路富集", method_text,
                           notes, warnings, outdir, params,
                           metrics={"n_features": len(_feature_list(ctx)), "n_significant": len(sig_features)},
                           store_extra={"significant": sig_features[:500]},
                           fig_title="Mummichog pathway enrichment")


def analysis_ora_kegg(ctx, params, outdir):
    """KEGG 通路 ORA：注释 → 通路集合，超几何 + BH（Kanehisa 2023）。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    de = _de_frame(ctx)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig = _significant(de, p_cut, params.get("fc"))
    if sig.empty:
        return _zero_significant_payload(
            ctx, params, outdir, title="KEGG 通路 ORA",
            method_text="内置实现（KEGG 通路超几何 + BH；依据 Kanehisa 2023）",
            fig_title="KEGG pathway ORA", p_cut=p_cut)
    library, _ = _load_library(ctx, params, notes, warnings)
    membership = _pathway_membership(ctx, library, notes)
    sig_features = sig["feature"].astype(str).tolist()
    frame = _ora(ctx, membership, "pathway", sig_features, params, notes)
    return _enrich_payload(frame, "pathway", "通路", "KEGG 通路 ORA",
                           "内置实现（KEGG 通路超几何 + BH；依据 Kanehisa 2023）",
                           notes, warnings, outdir, params,
                           metrics={"n_features": len(_feature_list(ctx)), "n_significant": len(sig_features)},
                           fig_title="KEGG pathway ORA")


def analysis_ora_hmdb_class(ctx, params, outdir):
    """HMDB 化学分类富集：class 注释超几何 + BH（Wishart 2022 HMDB 5.0）。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    de = _de_frame(ctx)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig = _significant(de, p_cut, params.get("fc"))
    if sig.empty:
        return _zero_significant_payload(
            ctx, params, outdir, title="HMDB 化学分类富集",
            method_text="内置实现（HMDB 化学分类超几何 + BH；依据 Wishart 2022）",
            label_key="class", label_title="化学分类", fig_title="HMDB chemical class enrichment",
            p_cut=p_cut, file_stem="hmdb-class")
    classes, basis = _class_membership(ctx, notes)
    membership = {feature: {label} for feature, label in classes.items() if label}
    sig_features = sig["feature"].astype(str).tolist()
    frame = _ora(ctx, membership, "class", sig_features, params, notes, min_size=int(params.get("min_size") or 3))
    payload = _enrich_payload(frame, "class", "化学分类", "HMDB 化学分类富集",
                              "内置实现（HMDB 化学分类超几何 + BH；依据 Wishart 2022）",
                              notes, warnings, outdir, params,
                              metrics={"n_features": len(_feature_list(ctx)), "n_significant": len(sig_features)},
                              file_stem="hmdb-class", fig_title="HMDB chemical class enrichment")
    payload["summary"]["bullets"].append(f"化学分类依据：{basis}")
    return payload


def analysis_metabolite_set_enrich(ctx, params, outdir):
    """MSEA：按 log2FC 排序的加权 KS 游走 + 置换（Subramanian 2005；Xia 2015）。"""
    params = dict(params or {})
    notes: list[str] = []
    warnings: list[str] = []
    de = _de_frame(ctx)
    if de.empty:
        raise prerequisite("差异表为空，无法做代谢物集富集")
    library, _ = _load_library(ctx, params, notes, warnings)
    membership = _pathway_membership(ctx, library, notes)
    if not any(membership.values()):
        raise prerequisite("特征表没有通路注释（pathway/kegg/代谢物名），无法做代谢物集富集")
    frame = _msea(ctx, de, membership, "pathway", params, notes)
    if frame.empty:
        raise prerequisite("没有满足最小集合大小（min_size）的通路，请补充注释或降低 min_size")
    extra_columns = [
        {"key": "es", "label": "ES", "type": "number"},
        {"key": "nes", "label": "NES", "type": "number"},
    ]
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    top_n = int(params.get("top_n") or 15)
    tables = [table("msea", "代谢物集富集（MSEA）结果",
                    _table_columns("pathway", "通路", extra_columns),
                    frame.to_dict(orient="records"),
                    note="内置实现（秩基加权 KS + 置换；依据 Subramanian 2005 / Xia 2015）")]
    figures = [
        _figure_bar(frame, outdir, "msea-bar", "MSEA: top metabolite sets", top_n=top_n, p_cut=p_cut),
        _figure_dot(frame, outdir, "msea-dot", "MSEA: set size vs significance", top_n=top_n),
    ]
    data_out: list[dict] = []
    try:
        path = export_csv(frame, outdir, "msea.csv", index=False)
        data_out.append(data_file("msea_csv", path, "MSEA 结果表（完整）"))
    except Exception as error:  # noqa: BLE001
        warnings.append(f"导出 MSEA 表失败：{error}")
    bullets = [
        "内置实现（MSEA 秩基加权 KS 游走 + 置换；依据 Subramanian 2005 / Xia 2015）",
        f"共检验 {len(frame)} 条集合，{int((frame['p_adj'] < p_cut).sum())} 条 FDR<{p_cut:g}",
    ]
    if not frame.empty:
        top = frame.iloc[0]
        bullets.append(
            f"最强集合：{top['pathway']}（ES={float(top['es']):.3f}，NES={float(top['nes']):.2f}，"
            f"FDR={float(top['p_adj']):.3g}）"
        )
    metrics = {"n_sets": int(len(frame)), "n_sig": int((frame["p_adj"] < p_cut).sum())}
    store = {"enrich": {"table": frame, "method": "MSEA", "label_key": "pathway",
                        "params": dict(params), "metrics": metrics}}
    return result(title="代谢物集富集（MSEA）", bullets=bullets, metrics=metrics, tables=tables,
                  figures=figures, warnings=warnings, notes=notes, data_out=data_out, store=store)


# ---------------------------------------------------------------------------
# 11. PLOTS（enrich_bar / enrich_dot）
# ---------------------------------------------------------------------------


def _resolve_enrich_frame(ctx, params):
    """图优先复用 results['enrich']['table']，没有就现场跑一次 Mummichog。"""
    results = ctx.get("results")
    if isinstance(results, dict):
        enrich = results.get("enrich")
        if isinstance(enrich, dict):
            frame = enrich.get("table")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                label_key = str(enrich.get("label_key") or "pathway")
                if label_key not in frame.columns:
                    label_key = "pathway" if "pathway" in frame.columns else frame.columns[0]
                return frame, label_key
            if isinstance(frame, list) and frame and isinstance(frame[0], dict):
                frame = pd.DataFrame(frame)
                return frame, "pathway"
    de = _de_frame(ctx)
    p_cut = float(params.get("p_cut") or DEFAULT_P_CUT)
    sig = _significant(de, p_cut)
    if sig.empty:
        raise prerequisite(f"当前阈值下没有显著差异特征，请先跑「单变量差异」或「通路富集」")
    notes: list[str] = []
    warnings: list[str] = []
    library, extra_masses = _load_library(ctx, params, notes, warnings)
    frame, status = _mummichog(ctx, sig, params, library, extra_masses, notes, warnings)
    if frame.empty:
        membership = _pathway_membership(ctx, library, notes)
        if not any(membership.values()):
            raise prerequisite("特征表没有通路注释、m/z 也没匹配到通路库，无法画富集图")
        frame = _ora(ctx, membership, "pathway", sig["feature"].astype(str).tolist(), params, notes)
    return frame, "pathway"


def plot_enrich_bar(ctx, params, outdir):
    """富集条形图（-log10 P，FDR<阈值的通路高亮）。"""
    params = dict(params or {})
    frame, label_key = _resolve_enrich_frame(ctx, params)
    if frame.empty:
        raise prerequisite("没有富集结果可以画图")
    return _figure_bar(frame, outdir, "enrich-bar", "Pathway enrichment: -log10(P)",
                       top_n=int(params.get("top_n") or 15),
                       p_cut=float(params.get("p_cut") or DEFAULT_P_CUT), label_key=label_key)


def plot_enrich_dot(ctx, params, outdir):
    """富集气泡图（x=命中比例，size=命中数，color=-log10 FDR）。"""
    params = dict(params or {})
    frame, label_key = _resolve_enrich_frame(ctx, params)
    if frame.empty:
        raise prerequisite("没有富集结果可以画图")
    return _figure_dot(frame, outdir, "enrich-dot", "Pathway enrichment: hit ratio",
                       top_n=int(params.get("top_n") or 15), label_key=label_key)


# ---------------------------------------------------------------------------
# 12. 模块导出（DESIGN §2.5）
# ---------------------------------------------------------------------------

STEPS = [
    {
        "id": "enrich",
        "title": "通路富集",
        "requires": ["de"],
        "params": [
            {"key": "method", "label": "方法", "type": "select", "default": "mummichog",
             "options": [{"value": "mummichog", "label": "Mummichog 式质量匹配 + 置换"},
                         {"value": "ora_kegg", "label": "KEGG 通路 ORA（注释超几何）"}]},
            {"key": "db", "label": "数据库", "type": "select", "default": "kegg"},
            {"key": "p_cut", "label": "P/FDR 阈值", "type": "number", "default": DEFAULT_P_CUT},
            {"key": "n_perm", "label": "置换次数", "type": "number", "default": DEFAULT_N_PERM},
            {"key": "ppm", "label": "质量精度(ppm)", "type": "number", "default": DEFAULT_PPM},
            {"key": "min_size", "label": "最小通路大小", "type": "number", "default": DEFAULT_MIN_SIZE},
        ],
        "run": step_enrich,
    },
]

ANALYSES = [
    {
        "id": "mummichog_pathway",
        "title": "Mummichog 式通路富集",
        "group": "通路与化学分类",
        "method": "质量映射 + 置换（Li 2013 mummichog；Uppal 2017；内置实现）",
        "needs": ["de"],
        "params": [
            {"key": "db", "label": "数据库", "type": "select", "default": "kegg"},
            {"key": "n_perm", "label": "置换次数", "type": "number", "default": DEFAULT_N_PERM},
            {"key": "ppm", "label": "质量精度(ppm)", "type": "number", "default": DEFAULT_PPM},
            {"key": "p_cut", "label": "P/FDR 阈值", "type": "number", "default": DEFAULT_P_CUT},
        ],
        "run": analysis_mummichog_pathway,
    },
    {
        "id": "ora_kegg",
        "title": "KEGG 通路 ORA",
        "group": "通路与化学分类",
        "method": "超几何 + BH（Kanehisa 2023 KEGG；内置实现）",
        "needs": ["de", "annotation"],
        "params": [
            {"key": "p_cut", "label": "P/FDR 阈值", "type": "number", "default": DEFAULT_P_CUT},
            {"key": "min_size", "label": "最小通路大小", "type": "number", "default": DEFAULT_MIN_SIZE},
        ],
        "run": analysis_ora_kegg,
    },
    {
        "id": "ora_hmdb_class",
        "title": "HMDB 化学分类富集",
        "group": "通路与化学分类",
        "method": "超几何 + BH（Wishart 2022 HMDB 5.0；内置实现）",
        "needs": ["de", "annotation"],
        "params": [
            {"key": "p_cut", "label": "P/FDR 阈值", "type": "number", "default": DEFAULT_P_CUT},
        ],
        "run": analysis_ora_hmdb_class,
    },
    {
        "id": "metabolite_set_enrich",
        "title": "代谢物集富集（MSEA）",
        "group": "通路与化学分类",
        "method": "秩基加权 KS + 置换（Subramanian 2005；Xia 2015；内置实现）",
        "needs": ["de"],
        "params": [
            {"key": "n_perm", "label": "置换次数", "type": "number", "default": DEFAULT_N_PERM},
            {"key": "min_size", "label": "最小集合大小", "type": "number", "default": 5},
        ],
        "run": analysis_metabolite_set_enrich,
    },
]


PLOTS = {
    "enrich_bar": plot_enrich_bar,
    "enrich_dot": plot_enrich_dot,
}


__all__ = ["STEPS", "ANALYSES", "PLOTS", "MB_PATHWAYS", "MB_METABOLITES",
           "MB_METABOLITE_MASS", "step_enrich"]
