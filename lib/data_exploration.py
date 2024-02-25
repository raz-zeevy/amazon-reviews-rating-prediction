import pandas as pd
from lib.utils import *

def load_data(path):
    if path == DATA_PATH:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, index_col=0)

##################
# Data Analysis  #
##################

def plot_brand_data(df):
    import pandas as pd
    import plotly.graph_objects as go

    # Group by 'brand' and aggregate
    grouped = df.groupby('brand').agg(
        unique_item_names=pd.NamedAgg(column='itemName', aggfunc='nunique'),
        total_rows=pd.NamedAgg(column='itemName', aggfunc='size'),
        unique_user_names=pd.NamedAgg(column='userName', aggfunc='nunique')
    )

    # Sort by 'unique_user_names' to divide the dataset
    grouped_sorted = grouped.sort_values(by='total_rows',
                                         ascending=False)

    # Divide the dataset
    top_brands = grouped_sorted.head(999)
    other_brands = grouped_sorted.iloc[999:]

    # Aggregate the two groups to get totals
    top_brands_totals = {
        'Total Unique Item Names': top_brands['unique_item_names'].sum(),
        'Total Rows': top_brands['total_rows'].sum(),
        'Total Unique User Names': top_brands['unique_user_names'].sum()
    }

    other_brands_totals = {
        'Total Unique Item Names': other_brands['unique_item_names'].sum(),
        'Total Rows': other_brands['total_rows'].sum(),
        'Total Unique User Names': other_brands['unique_user_names'].sum()
    }

    # Data for plotting
    categories = ['Total Unique Item Names', 'Total Rows',
                  'Total Unique User Names']
    top_brands_values = [top_brands_totals[category] for category in
                         categories]
    other_brands_values = [other_brands_totals[category] for category in
                           categories]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='Top 999 Brands', x=categories, y=top_brands_values),
        go.Bar(name='Other Brands', x=categories, y=other_brands_values)
    ])

    # Update layout
    fig.update_layout(barmode='group',
                      title='Comparison of Top 999 Brands vs Other '
                            'Brands (in terms of entries count)')
    fig.update_layout(
        title_font=dict(size=24),  # Increase title font size
        xaxis=dict(
            title='Category',
            title_font=dict(size=20),  # Increase x-axis title font size
            tickfont=dict(size=18)  # Increase x-axis tick font size
        ),
        yaxis=dict(
            title='Total',
            title_font=dict(size=20),  # Increase y-axis title font size
            tickfont=dict(size=18)  # Increase y-axis tick font size
        ),
        legend=dict(
            font=dict(
                size=18  # Increase legend font size
            )
        )
    )
    fig.show()


def correlation(df, col_a, col_b, method='pearson'):
    return df[col_a].corr(df[col_b],
                          method=method
                          )


def calculate_missing_values(df: pd.DataFrame):
    # Identify missing values in the dataset
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values for each column
    missing_percentage = (missing_values / len(df)) * 100

    missing_summary = pd.DataFrame(
        {'Missing Values': missing_values, 'Percentage': missing_percentage})
    missing_summary.sort_values(by="Percentage", ascending=False)
    return missing_summary


def plot_features_corr(df):
    global corr
    corrs = {}
    for col in ['featuresLength', 'featuresCount', 'price',
                'itemNameLength', 'itemNameCount', 'descriptionCount']:
        corr = correlation(df, col, 'rating', method='spearman')
        corrs[col] = corr
        print(f"Correlation between {col} and rating:"
              f" {corr:.2f}")
    # plot correlation graph for the features using plotly
    import plotly.express as px
    fig = px.bar(x=list(corrs.keys()), y=list(corrs.values()))
    fig.update_layout(title=f'Correlation between Features '
                            f'and Rating',
                      xaxis_title='Features',
                      yaxis_title='Correlation',
                      yaxis=dict(range=[-1, 1]))
    fig.show()

if __name__ == '__main__':
   pass
