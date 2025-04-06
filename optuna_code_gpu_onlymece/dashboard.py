import dash
from dash import dcc, html, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import glob

# Find the most recent results directory
def get_latest_results_dir():
    dirs = glob.glob("fraud_detection_results_*")
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)

def load_data(results_dir):
    """Load all necessary data files from the results directory"""
    data = {}
    
    # Load individual rules
    rules_path = os.path.join(results_dir, "optimized_rules.csv")
    if os.path.exists(rules_path):
        data['rules'] = pd.read_csv(rules_path)
    
    # Load ensemble rules
    ensemble_path = os.path.join(results_dir, "optimized_ensemble_rules.csv")
    if os.path.exists(ensemble_path):
        data['ensemble'] = pd.read_csv(ensemble_path)
    
    # Load optimization trials
    trials_path = os.path.join(results_dir, "optimization_trials.csv")
    if os.path.exists(trials_path):
        data['trials'] = pd.read_csv(trials_path)
    
    # Load summary
    summary_path = os.path.join(results_dir, "summary.csv")
    if os.path.exists(summary_path):
        data['summary'] = pd.read_csv(summary_path)
    
    # Load config
    config_path = os.path.join(results_dir, "best_config.csv")
    if os.path.exists(config_path):
        data['config'] = pd.read_csv(config_path)
    
    return data

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Fraud Detection Rules Dashboard", className="text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Results Directory"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='directory-dropdown',
                        options=[{'label': d, 'value': d} for d in glob.glob("fraud_detection_results_*")],
                        value=get_latest_results_dir(),
                        clearable=False
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Summary Statistics", className="bg-primary text-white"),
                dbc.CardBody(id="summary-stats")
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rule Performance", className="bg-info text-white"),
                dbc.CardBody([
                    dcc.Graph(id="rules-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Ensemble Rule Performance", className="bg-info text-white"),
                dbc.CardBody([
                    dcc.Graph(id="ensemble-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Optimization Trial Performance", className="bg-success text-white"),
                dbc.CardBody([
                    dcc.Graph(id="trials-chart")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Best Configuration", className="bg-warning text-white"),
                dbc.CardBody([
                    html.Div(id="config-table")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rule Details", className="bg-warning text-white"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='rule-dropdown',
                        placeholder='Select a rule to view details'
                    ),
                    html.Div(id="rule-details", className="mt-3")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
], fluid=True)

@callback(
    Output("summary-stats", "children"),
    Input("directory-dropdown", "value")
)
def update_summary_stats(results_dir):
    if not results_dir:
        return html.P("No results directory selected")
    
    data = load_data(results_dir)
    
    if 'summary' not in data:
        return html.P("Summary data not found")
    
    summary = data['summary']
    
    stats_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{summary['Total Rules'].iloc[0]}", className="text-center"),
                    html.P("Total Rules", className="text-center text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{summary['Total Ensemble Rules'].iloc[0]}", className="text-center"),
                    html.P("Total Ensemble Rules", className="text-center text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{summary['Detected Frauds (Individual)'].iloc[0]} / {summary['Total Frauds'].iloc[0]}", className="text-center"),
                    html.P("Detected Frauds", className="text-center text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{summary['Detection Rate (Individual)'].iloc[0]:.2%}", className="text-center"),
                    html.P("Detection Rate", className="text-center text-muted")
                ])
            ])
        ], width=3)
    ])
    
    return stats_cards

@callback(
    Output("rules-chart", "figure"),
    Input("directory-dropdown", "value")
)
def update_rules_chart(results_dir):
    if not results_dir:
        return go.Figure()
    
    data = load_data(results_dir)
    
    if 'rules' not in data:
        return go.Figure()
    
    rules = data['rules']
    
    # Sort rules by fraud count
    rules = rules.sort_values('Fraud_Count', ascending=False).head(15)
    
    # Create rule identifiers
    rules['Rule_Id'] = rules.apply(lambda row: f"Run {row['Run']} - Iter {row['Iteration']}", axis=1)
    
    # Create horizontal bar chart
    fig = px.bar(
        rules,
        y='Rule_Id',
        x='Fraud_Count',
        color='Fraud_Rate',
        orientation='h',
        color_continuous_scale='Viridis',
        hover_data=['Subset_Size', 'Fitness_Score'],
        title='Top 15 Individual Rules by Fraud Count'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig

@callback(
    Output("ensemble-chart", "figure"),
    Input("directory-dropdown", "value")
)
def update_ensemble_chart(results_dir):
    if not results_dir:
        return go.Figure()
    
    data = load_data(results_dir)
    
    if 'ensemble' not in data:
        return go.Figure(data=[go.Scatter(x=[0], y=[0], mode='markers')],
                         layout=go.Layout(title="No ensemble rules found"))
    
    ensemble = data['ensemble']
    
    # Sort ensemble rules by fraud count
    ensemble = ensemble.sort_values('Fraud_Count', ascending=False).head(10)
    
    # Create horizontal bar chart
    fig = px.bar(
        ensemble,
        y='Component_Rules',
        x='Fraud_Count',
        color='Fraud_Rate',
        orientation='h',
        color_continuous_scale='Viridis',
        hover_data=['Subset_Size'],
        title='Top 10 Ensemble Rules by Fraud Count'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig

@callback(
    Output("trials-chart", "figure"),
    Input("directory-dropdown", "value")
)
def update_trials_chart(results_dir):
    if not results_dir:
        return go.Figure()
    
    data = load_data(results_dir)
    
    if 'trials' not in data:
        return go.Figure()
    
    trials = data['trials']
    
    # Create scatter plot of trials
    fig = px.scatter(
        trials,
        x='Trial',
        y='Score',
        color='Fraud_Detection_Rate',
        size='Num_Rules',
        hover_data=['Avg_Fraud_Rate'],
        color_continuous_scale='Viridis',
        title='Optimization Trials Performance'
    )
    
    # Add a trendline to show progress
    fig.add_trace(
        go.Scatter(
            x=trials['Trial'],
            y=trials['Score'].rolling(window=3, min_periods=1).mean(),
            mode='lines',
            name='Rolling Average',
            line=dict(color='red', dash='dash')
        )
    )
    
    return fig

@callback(
    Output("config-table", "children"),
    Input("directory-dropdown", "value")
)
def update_config_table(results_dir):
    if not results_dir:
        return html.P("No results directory selected")
    
    data = load_data(results_dir)
    
    if 'config' not in data:
        return html.P("Configuration data not found")
    
    config = data['config']
    
    # Create a formatted table
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")])),
        html.Tbody([
            html.Tr([html.Td(row['Parameter']), html.Td(row['Value'])])
            for _, row in config.iterrows()
        ])
    ], striped=True, bordered=True, hover=True, size="sm")
    
    return table

@callback(
    Output("rule-dropdown", "options"),
    Output("rule-dropdown", "value"),
    Input("directory-dropdown", "value")
)
def update_rule_dropdown(results_dir):
    if not results_dir:
        return [], None
    
    data = load_data(results_dir)
    
    if 'rules' not in data:
        return [], None
    
    rules = data['rules']
    
    # Create options for dropdown
    options = [
        {'label': f"Run {row['Run']} - Iteration {row['Iteration']} (Fraud Count: {row['Fraud_Count']})",
         'value': i}
        for i, (_, row) in enumerate(rules.iterrows())
    ]
    
    return options, None

@callback(
    Output("rule-details", "children"),
    Input("rule-dropdown", "value"),
    Input("directory-dropdown", "value")
)
def update_rule_details(rule_idx, results_dir):
    if rule_idx is None or not results_dir:
        return html.P("Select a rule to view details")
    
    data = load_data(results_dir)
    
    if 'rules' not in data:
        return html.P("Rules data not found")
    
    rules = data['rules']
    rule = rules.iloc[rule_idx]
    
    # Parse rule conditions
    rule_conditions = []
    for condition in rule['Rule'].split(';'):
        if condition.strip():
            rule_conditions.append(condition.strip())
    
    # Create the rule details card
    details = html.Div([
        html.H5(f"Rule from Run {rule['Run']}, Iteration {rule['Iteration']}"),
        html.P(f"Fraud Count: {rule['Fraud_Count']}"),
        html.P(f"Fraud Rate: {rule['Fraud_Rate']:.4%}"),
        html.P(f"Subset Size: {rule['Subset_Size']}"),
        html.P(f"Fitness Score: {rule['Fitness_Score']:.4f}"),
        html.H6("Rule Conditions:"),
        html.Ul([html.Li(condition) for condition in rule_conditions])
    ])
    
    return details

if __name__ == '__main__':
    app.run(debug=True)