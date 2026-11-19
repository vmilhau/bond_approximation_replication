#%%
from IPython.core.getipython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

# ===== IMPORTS =====

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
from pathlib import Path
import paper_analysis as pa

plt.rcParams.update({"axes.grid": True})

plt.rcParams.update({
    "axes.grid": True,

    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,    
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ===== CONFIGURATION =====

THRESHOLD_2ND_ORDER = 0.05  # Maximum second-order error for a bond to be plotted
SAMPLE_PERIODS = {'Daily': 1, 'Monthly': 20, 'Annual': 240}  # Sample periods in business days
OUTPUT_SUBDIR = 'Mock results'


# ===== SETUP: Load and validate input files =====
# Define file paths
base_dir = Path(__file__).resolve().parent
dsc_file_path = base_dir / 'amtdsc1.fil'
prc_file_path = base_dir / 'amtprc1.fil'
typ_file_path = base_dir / 'amttyp.csv'
auc_file_path = base_dir / 'v1-accounting-od-auctions_query.csv'
rat_file_path = base_dir / 'DGS1.csv'
spr_file_path = base_dir / 'T10YFF.csv'


# Validate file existence
for p in [dsc_file_path, prc_file_path, typ_file_path, auc_file_path, rat_file_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

bond_dsc_data = pa.read_bond_dsc(dsc_file_path)

bond_prc_data = (pa.read_bond_prc(prc_file_path)
                 .set_index(['Identifier', 'Pricing Date'])  # MultiIndex for easier access
                 .sort_index()  # Lexicographically sort on Identifier and Pricing Date
                 )

bond_typ_data = (pa.read_bond_typ(typ_file_path)
                 .set_index('Identifier')
                 .sort_index()  # Lexicographically sort on Identifier and Pricing Date
                 )

bond_auc_data = pa.read_bond_auction_data(auc_file_path)

raw_bond_data = pa.RawBondData(bond_dsc_data=bond_dsc_data, bond_prc_data=bond_prc_data, 
                            bond_typ_data=bond_typ_data, bond_auc_data=bond_auc_data)

raw_rat_data = pa.read_rate_data(rat_file_path)

raw_spr_data = pa.read_rate_data(spr_file_path)

paper_raw_data = pa.PaperRawData(raw_bond_data, raw_rat_data, raw_spr_data)


# ===== PROCESSING: Filter, clean, and calculate bond metrics =====


#%% Cleaning and filtering data
t0 = time.time()
bond_data = pa.filter_and_clean(paper_raw_data.raw_bond_data)
t1 = time.time()
print(f"Elapsed time - Data cleaning and filtering: {t1-t0:.3f}s")


#%% Calculating YTM, durations, convexities, accrued interests and dirty prices
t0 = time.time()
calculated_bond_data = pa.bond_calculation(bond_data)
t1 = time.time()
print(f"Elapsed time - Bond calculations: {t1-t0:.3f}s")


#%% Calculating return contributions, actual total returns and approximate total returns
t0 = time.time()
dict_analysis_results = pa.analysis_by_frequency(calculated_bond_data, SAMPLE_PERIODS)
t1 = time.time()
print(f"Elapsed time - Return analysis: {t1-t0:.3f}s")


# ===== OUTPUT: Generate plots and save statistics =====
#%% Plot actual vs approximate total returns
daily_tr = dict_analysis_results['Daily'].total_returns
is_plotted = (daily_tr['Actual TR'] - daily_tr['2nd Order']).abs() <= THRESHOLD_2ND_ORDER
plotted_index = is_plotted[is_plotted].index
dict_results_filtered = {k: v.subset(plotted_index, level='Identifier') for k, v in dict_analysis_results.items()}


for m, w in dict_results_filtered.items():
    scatter_plot_data = w.scatter_plot_data()
    
    output_dir = base_dir / OUTPUT_SUBDIR / m
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    pa.scatter_plot_returns(scatter_plot_data, output_dir)


#%% Save statistics:
# - bond data
# - clean prices conditional on pricing errors
# - contributions
output_dir = base_dir / OUTPUT_SUBDIR
output_dir.mkdir(parents=True, exist_ok=True)

description = bond_data.description()

description.to_csv(output_dir / 'Bond_description.csv')

print("Saved Bond_description.csv")


small_error_identifiers = is_plotted[is_plotted].index
large_error_identifiers = is_plotted[~is_plotted].index

clean_prices_stats = dict_analysis_results['Daily'].clean_prices_stats(small_error_identifiers, large_error_identifiers)

large_error_summary = clean_prices_stats['Large error summary']
small_error_summary = clean_prices_stats['Small error summary']

output_dir = base_dir / OUTPUT_SUBDIR / 'Daily'
output_dir.mkdir(parents=True, exist_ok=True)

large_error_summary.to_csv(output_dir / 'Large_error_summary.csv')
print("Saved Large_error_summary.csv")

small_error_summary.to_csv(output_dir / 'Small_error_summary.csv')
print("Saved Small_error_summary.csv")


output_dir = base_dir / OUTPUT_SUBDIR / 'Daily'
output_dir.mkdir(parents=True, exist_ok=True)

approximation_stats = dict_analysis_results['Daily'].subset(small_error_identifiers, level='Identifier').approximation_stats()

approximation_stats.to_csv(output_dir / 'Approximation_stats.csv')
print("Saved Approximation_stats.csv")


# %%
output_dir = base_dir / OUTPUT_SUBDIR / 'Daily'
output_dir.mkdir(parents=True, exist_ok=True)
pa.scatter_plot_income_subperiods(calculated_bond_data, output_dir)


# %%
paper_rate_data = paper_raw_data.paper_rate_data()
level_data = paper_rate_data.level_data
level_regime = pa.quantile_regime(level_data.data, 3, ['low', 'medium', 'high'])
options = pa.PlotOptions(
    yaxis_formatter=mtick.PercentFormatter(1.)
)
level_regime.plot(level_data.data, options=options, output_fil=output_dir / 'level_regime.pdf')

volatility_data = paper_rate_data.volatility_data
vol_regime = pa.quantile_regime(volatility_data.data, 3, ['low', 'medium', 'high'])
vol_regime.plot(volatility_data.data, output_fil=output_dir / 'volatility_regime.pdf')


# %%
d_level = pa.averages_by_regime(level_regime, dict_results_filtered['Daily'])
d_vol = pa.averages_by_regime(vol_regime, dict_results_filtered['Daily'])

panel_labels = {
    "average_yield": "Yield (in \\%)",
    "average_duration": "Duration",
    "average_convexity": "Convexity",
    "average_income_contribution": "Income contribution ($\\times10^3$)",
    "average_duration_contribution": "Absolute duration contribution ($\\times10^4$)",
    "average_convexity_contribution": "Convexity contribution ($\\times10^6$)",
}

pa.averages_by_two_regimes_to_latex(
    left_results=d_level,
    right_results=d_vol,
    left_label="Level regime",
    right_label="Volatility regime",
    panel_labels=panel_labels,
    path=output_dir / "averages_by_regime.tex",
)


# %%
average_diff_by_vol_regime = (vol_regime
                              .diff_2nd_vs_1st_in_subperiods(dict_results_filtered['Daily'])
                              .average_by_regime()
                              .rename_axis('Volatility regime', axis=1)
                              )

pa.by_regime_2_levels_to_latex(1e4 * average_diff_by_vol_regime, output_dir / 'diff_2nd_vs_1st_by_volatility.tex')


# %%
spread_data = paper_rate_data.spread_data
spread_regime = pa.quantile_regime(spread_data.drop_na().data, 3, ['low', 'medium', 'high'])
average_diff_by_spread_regime = (spread_regime
                              .diff_2nd_vs_1st_in_subperiods(dict_results_filtered['Daily'])
                              .average_by_regime()
                              .rename_axis('Level regime', axis=1)
                              )
pa.by_regime_2_levels_to_latex(1e4 * average_diff_by_spread_regime, output_dir / 'diff_2nd_vs_1st_by_spread.tex')

average_diff_by_spread_regime = (spread_regime
                              .diff_2nd_vs_2nd_wo_income_in_subperiods(dict_results_filtered['Daily'])
                              .average_by_regime()
                              .rename_axis('Level regime', axis=1)
                              )
pa.by_regime_2_levels_to_latex(1e4 * average_diff_by_spread_regime, output_dir / 'diff_2nd_vs_2nd_wo_income_by_spread.tex')
