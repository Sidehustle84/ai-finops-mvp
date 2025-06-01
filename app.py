# AI FinOps MVP - Simple Cost Tracking Dashboard
# This is a minimal viable product for tracking AI/LLM costs across providers

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="AI FinOps - Cost Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIProviderTracker:
    """Track costs across different AI providers"""
    
    def __init__(self):
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'api_base': 'https://api.openai.com/v1',
                'models': {
                    'gpt-4': {'input': 0.00003, 'output': 0.00006},
                    'gpt-4-turbo': {'input': 0.00001, 'output': 0.00003},
                    'gpt-3.5-turbo': {'input': 0.0000005, 'output': 0.0000015}
                }
            },
            'anthropic': {
                'name': 'Anthropic',
                'models': {
                    'claude-3-opus': {'input': 0.000015, 'output': 0.000075},
                    'claude-3-sonnet': {'input': 0.000003, 'output': 0.000015},
                    'claude-3-haiku': {'input': 0.00000025, 'output': 0.00000125}
                }
            },
            'google': {
                'name': 'Google AI',
                'models': {
                    'gemini-pro': {'input': 0.0000005, 'output': 0.0000015},
                    'gemini-ultra': {'input': 0.00001, 'output': 0.00003}
                }
            }
        }
    
    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a specific provider/model combination"""
        if provider not in self.providers or model not in self.providers[provider]['models']:
            return 0.0
        
        pricing = self.providers[provider]['models'][model]
        cost = (input_tokens * pricing['input']) + (output_tokens * pricing['output'])
        return round(cost, 6)
    
    def get_openai_usage(self, api_key: str, days: int = 30) -> List[Dict]:
        """Fetch actual OpenAI usage data (placeholder - requires API implementation)"""
        # This would connect to OpenAI's usage API in a real implementation
        # For MVP, we'll simulate some data
        import random
        
        usage_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            daily_requests = random.randint(50, 300)
            
            for _ in range(daily_requests):
                model = random.choice(['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'])
                input_tokens = random.randint(100, 2000)
                output_tokens = random.randint(50, 1000)
                cost = self.calculate_cost('openai', model, input_tokens, output_tokens)
                
                usage_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'provider': 'openai',
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost,
                    'project': random.choice(['chatbot', 'content-gen', 'analysis', 'support'])
                })
        
        return usage_data

def main():
    st.title("🤖💰 AI FinOps - Cost Tracker MVP")
    st.markdown("Track and optimize your AI/LLM spending across providers")
    
    # Initialize tracker
    tracker = AIProviderTracker()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Keys (in real app, these would be encrypted)
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
    anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password", help="Your Anthropic API key")
    
    # Date range
    days_to_fetch = st.sidebar.slider("Days of data to analyze", 7, 90, 30)
    
    # Mock data toggle for demo
    use_mock_data = st.sidebar.checkbox("Use demo data", value=True, help="Toggle for demonstration")
    
    if use_mock_data:
        st.sidebar.info("Using simulated data for demonstration")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate or fetch data
    if use_mock_data:
        usage_data = tracker.get_openai_usage("demo_key", days_to_fetch)
        df = pd.DataFrame(usage_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate metrics
        total_cost = df['cost'].sum()
        total_requests = len(df)
        avg_daily_cost = total_cost / days_to_fetch
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        
        # Display key metrics
        with col1:
            st.metric("Total Cost", f"${total_cost:.2f}", f"${avg_daily_cost:.2f}/day")
        
        with col2:
            st.metric("Total Requests", f"{total_requests:,}", f"{total_requests//days_to_fetch}/day avg")
        
        with col3:
            st.metric("Cost per Request", f"${cost_per_request:.4f}")
        
        with col4:
            monthly_projection = avg_daily_cost * 30
            st.metric("Monthly Projection", f"${monthly_projection:.2f}")
        
        # Charts
        st.subheader("📊 Cost Analysis")
        
        # Daily cost trend
        col1, col2 = st.columns(2)
        
        with col1:
            daily_costs = df.groupby('date')['cost'].sum().reset_index()
            fig_trend = px.line(daily_costs, x='date', y='cost', 
                              title='Daily Cost Trend',
                              labels={'cost': 'Cost ($)', 'date': 'Date'})
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Cost by model
            model_costs = df.groupby('model')['cost'].sum().reset_index()
            fig_model = px.pie(model_costs, values='cost', names='model', 
                             title='Cost Distribution by Model')
            st.plotly_chart(fig_model, use_container_width=True)
        
        # Cost by project
        col1, col2 = st.columns(2)
        
        with col1:
            project_costs = df.groupby('project')['cost'].sum().reset_index()
            fig_project = px.bar(project_costs, x='project', y='cost',
                               title='Cost by Project/Use Case',
                               labels={'cost': 'Cost ($)', 'project': 'Project'})
            st.plotly_chart(fig_project, use_container_width=True)
        
        with col2:
            # Token usage efficiency
            df['total_tokens'] = df['input_tokens'] + df['output_tokens']
            df['cost_per_token'] = df['cost'] / df['total_tokens']
            efficiency = df.groupby('model')['cost_per_token'].mean().reset_index()
            fig_efficiency = px.bar(efficiency, x='model', y='cost_per_token',
                                  title='Cost Efficiency by Model ($/token)',
                                  labels={'cost_per_token': 'Cost per Token ($)'})
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Optimization recommendations
        st.subheader("💡 Cost Optimization Recommendations")
        
        # Calculate potential savings
        gpt4_usage = df[df['model'] == 'gpt-4']
        gpt4_cost = gpt4_usage['cost'].sum()
        
        if gpt4_cost > 0:
            # Simulate savings by switching some GPT-4 to GPT-4-turbo
            potential_savings = gpt4_cost * 0.3  # 30% savings
            st.success(f"💰 **Potential Savings: ${potential_savings:.2f}/month**")
            st.write("• Consider using GPT-4-turbo for 70% of your GPT-4 use cases (3x cheaper)")
            st.write("• Implement response caching for repeated queries")
            st.write("• Use GPT-3.5-turbo for simple tasks where possible")
        
        # Recent usage table
        st.subheader("📋 Recent Usage Details")
        recent_data = df.head(100).sort_values('date', ascending=False)
        st.dataframe(recent_data[['date', 'provider', 'model', 'project', 'input_tokens', 'output_tokens', 'cost']], 
                    use_container_width=True)
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"ai_costs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate summary report
            report = f"""
AI Cost Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Period: {days_to_fetch} days

Total Cost: ${total_cost:.2f}
Total Requests: {total_requests:,}
Average Daily Cost: ${avg_daily_cost:.2f}
Monthly Projection: ${monthly_projection:.2f}

Top Models by Cost:
{model_costs.sort_values('cost', ascending=False).to_string(index=False)}

Recommendations:
- Monitor daily spending trends
- Consider model optimization for cost efficiency
- Implement usage quotas by project
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"ai_cost_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    else:
        st.info("Enter your API keys in the sidebar to fetch real usage data")
        st.markdown("""
        **To get started:**
        1. Add your OpenAI API key in the sidebar
        2. Optionally add Anthropic and Google AI keys
        3. Select the date range for analysis
        4. View your cost breakdown and optimization recommendations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Want to optimize your AI costs?** This MVP shows basic tracking. Full version includes advanced analytics, alerts, and automated optimization.")

if __name__ == "__main__":
    main()
