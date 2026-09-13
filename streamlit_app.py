import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

st.set_page_config(page_title="Olist EDA", layout="wide")
st.title("Olist EDA")

@st.cache_data
def load_data():
	return pd.read_csv("olist.csv")


def format_labels(labels, short=False):
	formatted = []
	for label in labels:
		spaced = " ".join(str(label).split("_"))
		if len(spaced) > 20 and short:
			formatted.append(spaced[:18] + "...")
		else:
			formatted.append("\n".join(wrap(spaced, 20)))
	return formatted


def display_figure(fig):
	fig.tight_layout()
	st.pyplot(fig, use_container_width=True)
	plt.close(fig)


def annotate_bars(ax, values, percentages=None, decimals=1):
	visible_patches = [patch for patch in ax.patches if patch.get_height() > 0]
	for index, (patch, value) in enumerate(zip(visible_patches, values)):
		if patch.get_height() <= 0:
			continue
		label = f"{int(value):,}"
		if percentages is not None:
			label += f" ({percentages[index]:.{decimals}f}%)"
		ax.annotate(
			label,
			(patch.get_x() + patch.get_width() / 2, patch.get_height()),
			ha="center",
			va="bottom",
			fontsize=7,
		)


def score_class_proportions(dataframe, column):
	proportions = pd.crosstab(
		dataframe[column], dataframe["score_class"], normalize="index"
	)
	proportions = proportions.loc[dataframe[column].value_counts().index]
	counts = pd.crosstab(dataframe[column], dataframe["score_class"]).reindex(
		index=proportions.index, columns=proportions.columns, fill_value=0
	)
	return proportions, counts


def plot_proportion_bars(ax, proportions, counts):
	proportions.plot(kind="bar", stacked=True, colormap="Set2", ax=ax)
	for container, category in zip(ax.containers, proportions.columns):
		for patch, percentage, count in zip(
			container, proportions[category], counts[category]
		):
			if percentage > 0.03:
				ax.text(
					patch.get_x() + patch.get_width() / 2,
					patch.get_y() + patch.get_height() / 2,
					f"{percentage:.1%}\n({count:,})",
					ha="center",
					va="center",
					fontsize=8,
				)


def plot_average_scores(dataframe, column, title, palette="viridis"):
	scores = dataframe.groupby(column)["review_score"].mean().sort_values()
	fig, ax = plt.subplots(figsize=(8, max(3, len(scores) * 0.35)))
	sns.barplot(x=scores.values, y=format_labels(scores.index), palette=palette, ax=ax)
	for bar, score in zip(ax.patches, scores.values):
		ax.text(
			score - 0.05,
			bar.get_y() + bar.get_height() / 2,
			f"{score:.2f}",
			ha="right",
			va="center",
			color="white",
		)
	ax.set_title(title)
	ax.set_xlabel("Average Review Score")
	ax.set_ylabel(format_labels([column])[0])
	return fig


data = load_data()
sns.set_theme(style="whitegrid")

st.header("Review Score")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, column, title in zip(
	axes,
	["review_score", "score_class"],
	["Review score distribution", "Review score category distribution"],
):
	counts = data[column].value_counts().sort_index()
	sns.countplot(
		data=data,
		x=column,
		order=counts.index,
		hue=column,
		legend=False,
		palette="viridis",
		ax=ax,
	)
	percentages = counts / counts.sum() * 100
	for patch, count, percentage in zip(ax.patches, counts, percentages):
		ax.annotate(
			f"{count:,} ({percentage:.1f}%)",
			(patch.get_x() + patch.get_width() / 2, patch.get_height()),
			ha="center",
			va="bottom",
			fontsize=7,
		)
	ax.set_title(title)
	ax.set_xlabel(format_labels([column])[0])
	ax.set_ylabel("Count")
display_figure(fig)

# st.subheader("Customer Offset")
# fig, ax = plt.subplots(figsize=(8, 3))
# sns.boxplot(data=data, x="customer_offset", y="review_score", ax=ax)
# ax.set_title("Review Score Distribution by Customer ETA Offset")
# ax.set_xlabel("Offset days from customer ETA")
# ax.set_ylabel("Review Score")
# display_figure(fig)

st.header("Numerical Features")
selected_columns = [
	"review_score",
	"price",
	"freight_value",
	"product_photos_qty",
	"payment_sequential",
	"payment_installments_total",
	"carrier_offset",
	"customer_offset",
	"delivered_carrier_days",
	"delivered_customer_days",
	"approval_duration",
	"customer_eta_days",
]

fig, ax = plt.subplots(figsize=(9, 7))
selected_corr = data[selected_columns].corr()
sns.heatmap(
	selected_corr,
	cmap=sns.diverging_palette(230, 20, as_cmap=True),
	annot=True,
	fmt=".2f",
	mask=np.triu(np.ones_like(selected_corr, dtype=bool)),
	square=True,
	annot_kws={"size": 7},
	xticklabels=format_labels(selected_columns, short=True),
	yticklabels=format_labels(selected_columns, short=True),
	ax=ax,
)
ax.set_title("Correlation Matrix of Selected Numerical Features")
display_figure(fig)

fig, ax = plt.subplots(figsize=(11, 8))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(
	numeric_data.corr(),
	cmap=sns.diverging_palette(230, 20, as_cmap=True),
	annot=True,
	fmt=".2f",
	square=True,
	annot_kws={"size": 6},
	xticklabels=format_labels(numeric_data.columns, short=True),
	yticklabels=format_labels(numeric_data.columns, short=True),
	ax=ax,
)
ax.set_title("Heatmap of All Numerical Features")
display_figure(fig)

st.header("Average Review Scores")
st.pyplot(plot_average_scores(data, "product_category_name_english", "Average Review Score by Product Category"), use_container_width=True)
plt.close("all")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for ax, column, title in zip(
	axes,
	["customer_state", "seller_state"],
	["Customer States", "Seller States"],
):
	scores = data.groupby(column)["review_score"].mean()
	top = scores.nlargest(10).sort_values()
	bottom = scores.nsmallest(10).sort_values()
	combined = pd.concat([top, bottom]).drop_duplicates().sort_values()
	sns.barplot(x=combined.values, y=format_labels(combined.index), ax=ax)
	ax.set_title(f"Top and Bottom States: {title}")
	ax.set_xlabel("Average Review Score")
	ax.set_ylabel("State")
display_figure(fig)

display_figure(plot_average_scores(data, "order_status", "Average Review Score by Order Status"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, column, title in zip(
	axes,
	["delivered_carrier", "delivered_customer"],
	["Carrier Delivery Status", "Customer Delivery Status"],
):
	scores = data.groupby(column)["review_score"].mean().sort_values()
	sns.barplot(x=scores.values, y=format_labels(scores.index), palette="ocean", ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Average Review Score")
	ax.set_ylabel("Delivery Status")
display_figure(fig)

st.header("Review Text Availability")
empty_reviews = pd.DataFrame(
	{
		"review_score": data["review_score"],
		"empty_comment": data["review_comment_message"].isna(),
		"empty_title": data["review_comment_title"].isna(),
	}
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, column, title, palette in zip(
	axes,
	["empty_comment", "empty_title"],
	["Distribution of Reviews with Comments", "Distribution of Reviews with Titles"],
	["ocean", "viridis"],
):
	sns.countplot(
		data=empty_reviews,
		x="review_score",
		hue=column,
		palette=palette,
		ax=ax,
	)
	ax.set_title(title)
	ax.set_xlabel("Review Score")
	ax.set_ylabel("Count")
	ax.legend(labels=["Filled", "Empty"])
display_figure(fig)

empty_reviews_cat = pd.DataFrame(
	{
		"score_class": data["score_class"],
		"empty_comment": data["review_comment_message"].isna(),
		"empty_title": data["review_comment_title"].isna(),
	}
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, column, title, palette in zip(
	axes,
	["empty_comment", "empty_title"],
	["Comments by Score Class", "Titles by Score Class"],
	["ocean", "viridis"],
):
	sns.countplot(
		data=empty_reviews_cat,
		x="score_class",
		order=["Low", "High"],
		hue=column,
		palette=palette,
		ax=ax,
	)
	ax.set_title(title)
	ax.set_xlabel("Score Class")
	ax.set_ylabel("Count")
	ax.legend(labels=["Filled", "Empty"])
display_figure(fig)

st.header("Score Class Proportions")
for column, title in [
	("delivered_carrier", "Score Class Proportions by Carrier Delivery Status"),
	("delivered_customer", "Score Class Proportions by Customer Delivery Status"),
	("seller_state", "Score Class Proportions by Top Seller States"),
	("customer_state", "Score Class Proportions by Top Customer States"),
	("order_status", "Score Class Proportions by Order Status"),
]:
	proportions, counts = score_class_proportions(data, column)
	proportions = proportions.head(10)
	counts = counts.reindex(proportions.index)
	fig, ax = plt.subplots(figsize=(10, 5))
	plot_proportion_bars(ax, proportions, counts)
	ax.set_title(title)
	ax.set_xlabel(format_labels([column])[0])
	ax.set_ylabel("Proportion")
	ax.set_ylim(0, 1)
	ax.set_xticks(range(len(proportions.index)))
	ax.set_xticklabels(format_labels(proportions.index), rotation=45, ha="right")
	ax.legend(title="Score Class", bbox_to_anchor=(1, 1), loc="upper left")
	display_figure(fig)

category_scores = data.groupby("product_category_name_english")["review_score"].mean()
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
for ax, categories, title in zip(
	axes,
	[category_scores.nlargest(10).sort_values(), category_scores.nsmallest(10).sort_values()],
	["Top 10 Product Categories by Score Class", "Bottom 10 Product Categories by Score Class"],
):
	proportions, counts = score_class_proportions(data, "product_category_name_english")
	proportions = proportions.reindex(categories.index)
	counts = counts.reindex(proportions.index)
	proportions.plot(kind="barh", stacked=True, colormap="Set2", ax=ax, legend=False)
	for container, category in zip(ax.containers, proportions.columns):
		for patch, percentage, count in zip(container, proportions[category], counts[category]):
			if percentage > 0.03:
				ax.text(
					patch.get_x() + patch.get_width() / 2,
					patch.get_y() + patch.get_height() / 2,
					f"{percentage:.1%}\n({count:,})",
					ha="center",
					va="center",
					fontsize=8,
				)
	ax.set_title(title)
	ax.set_xlim(0, 1)
	ax.set_xlabel("Proportion")
	ax.set_ylabel("Product Category")
	ax.set_yticks(range(len(categories)))
	ax.set_yticklabels(format_labels(categories.index))
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Score Class", loc="center left", bbox_to_anchor=(1, 0.5))
display_figure(fig)
