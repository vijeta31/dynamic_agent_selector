# langgraph_dynamic_agents.py
import asyncio
from typing import List, Dict, Any, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver
import operator

# Define the state that flows through the graph
class AgentWorkflowState(TypedDict):
    query: str
    planner_output: List[str]
    current_step: int
    agent_results: Dict[str, str]
    accumulated_context: str
    final_answer: str
    workflow_complete: bool
    execution_path: List[str]

class DynamicAgentFlowLangGraph:
    """Dynamic Agent Flow Builder using LangGraph"""
    
    def __init__(self, llm):
        self.llm = llm
        self.graphs: Dict[str, StateGraph] = {}
        self._initialize_agent_functions()
    
    def _initialize_agent_functions(self):
        """Initialize agent functions as LangGraph nodes"""
        
        # Each agent is a node function that processes state
        self.agent_functions = {
            "agent_purchase": self._purchase_agent_node,
            "agent_account": self._account_agent_node, 
            "agent_inventory": self._inventory_agent_node,
            "agent_support": self._support_agent_node,
            "agent_analytics": self._analytics_agent_node,
            "agent_generate_insights": self._insights_agent_node
        }
    
    async def _purchase_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Purchase Agent Node"""
        
        print(f"🛒 Executing Purchase Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are a Purchase Analysis Agent. Analyze the customer's purchase history.
        
        Query: {state['query']}
        Previous Context: {state['accumulated_context']}
        
        Provide detailed purchase analysis including:
        - Recent purchase history
        - Purchase patterns and trends  
        - Transaction amounts and frequency
        - Key purchase insights
        
        Focus specifically on purchase-related aspects of the query.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_purchase"] = result
        state["accumulated_context"] += f"\n\nPurchase Agent Results:\n{result}"
        state["execution_path"].append("agent_purchase")
        state["current_step"] += 1
        
        return state
    
    async def _account_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Account Agent Node"""
        
        print(f"👤 Executing Account Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are an Account Management Agent. Analyze the customer's account status.
        
        Query: {state['query']}
        Previous Context: {state['accumulated_context']}
        
        Provide account analysis including:
        - Customer profile and membership status
        - Account health and payment history
        - Loyalty metrics and engagement
        - Account-related insights
        
        Build upon previous agent results if available.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_account"] = result  
        state["accumulated_context"] += f"\n\nAccount Agent Results:\n{result}"
        state["execution_path"].append("agent_account")
        state["current_step"] += 1
        
        return state
    
    async def _inventory_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Inventory Agent Node"""
        
        print(f"📦 Executing Inventory Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are an Inventory Management Agent. Check product availability and stock status.
        
        Query: {state['query']}
        Previous Context: {state['accumulated_context']}
        
        Provide inventory information including:
        - Current stock levels and availability
        - Product inventory status
        - Stock alerts and forecasts
        - Availability for customer requests
        
        Give direct, actionable inventory information.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_inventory"] = result
        state["accumulated_context"] += f"\n\nInventory Agent Results:\n{result}"
        state["execution_path"].append("agent_inventory")
        state["current_step"] += 1
        
        return state
    
    async def _support_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Support Agent Node"""
        
        print(f"🎧 Executing Support Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are a Customer Support Agent. Analyze customer service interactions.
        
        Query: {state['query']}
        Previous Context: {state['accumulated_context']}
        
        Provide support analysis including:
        - Support ticket history and resolutions
        - Customer satisfaction ratings
        - Service interaction patterns
        - Support quality metrics
        
        Contribute support insights to the overall customer analysis.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_support"] = result
        state["accumulated_context"] += f"\n\nSupport Agent Results:\n{result}"
        state["execution_path"].append("agent_support")
        state["current_step"] += 1
        
        return state
    
    async def _analytics_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Analytics Agent Node"""
        
        print(f"📊 Executing Analytics Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are a Customer Analytics Agent. Generate behavioral insights and patterns.
        
        Query: {state['query']}
        Previous Context: {state['accumulated_context']}
        
        Provide analytics including:
        - Customer engagement patterns
        - Behavioral trends and lifecycle analysis
        - Predictive insights
        - Data-driven customer metrics
        
        Analyze patterns from all available previous agent data.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_analytics"] = result
        state["accumulated_context"] += f"\n\nAnalytics Agent Results:\n{result}"
        state["execution_path"].append("agent_analytics")
        state["current_step"] += 1
        
        return state
    
    async def _insights_agent_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Insights Agent Node - Synthesis and Final Recommendations"""
        
        print(f"💡 Executing Insights Agent (Step {state['current_step']})")
        
        prompt = f"""
        You are a Strategic Insights Agent. Synthesize ALL previous agent results into comprehensive insights.
        
        Original Query: {state['query']}
        
        Complete Context from Previous Agents:
        {state['accumulated_context']}
        
        Agent Results Summary:
        {chr(10).join([f"- {agent}: {result[:150]}..." for agent, result in state['agent_results'].items()])}
        
        Generate comprehensive final insights that:
        1. Synthesize information from ALL previous agents
        2. Answer the original query completely
        3. Provide actionable recommendations
        4. Create strategic conclusions
        
        This is the final step - provide complete, synthesized insights.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content
        
        # Update state
        state["agent_results"]["agent_generate_insights"] = result
        state["final_answer"] = result
        state["execution_path"].append("agent_generate_insights")
        state["current_step"] += 1
        state["workflow_complete"] = True
        
        return state
    
    def _create_router_function(self, planner_output: List[str]):
        """Create dynamic routing function based on planner output"""
        
        def route_next_agent(state: AgentWorkflowState) -> str:
            """Dynamic router that follows planner output sequence"""
            
            current_step = state["current_step"]
            
            # Check if we've completed all planned agents
            if current_step >= len(planner_output):
                return END
            
            # Get next agent from planner output
            next_agent = planner_output[current_step]
            
            print(f"🔀 Router: Step {current_step} -> {next_agent}")
            return next_agent
        
        return route_next_agent
    
    def build_dynamic_graph(self, planner_output: List[str]) -> StateGraph:
        """Build dynamic graph based on planner output"""
        
        print(f"🔧 Building Dynamic Graph for: {planner_output}")
        
        # Create new state graph
        workflow = StateGraph(AgentWorkflowState)
        
        # Add all potential agent nodes
        for agent_name in planner_output:
            if agent_name in self.agent_functions:
                workflow.add_node(agent_name, self.agent_functions[agent_name])
                print(f"➕ Added node: {agent_name}")
        
        # Create router function for this specific planner output
        router = self._create_router_function(planner_output)
        
        # Set entry point
        workflow.set_entry_point(planner_output[0])
        
        # Add conditional edges for dynamic routing
        for i, agent_name in enumerate(planner_output):
            if i < len(planner_output) - 1:
                # Not the last agent - route to next
                next_agent = planner_output[i + 1]
                workflow.add_edge(agent_name, next_agent)
            else:
                # Last agent - end workflow
                workflow.add_edge(agent_name, END)
        
        return workflow
    
    async def execute_dynamic_workflow(self, query: str, planner_output: List[str]) -> Dict[str, Any]:
        """Main execution method - builds and runs dynamic workflow"""
        
        print(f"\n🚀 LANGGRAPH DYNAMIC WORKFLOW")
        print(f"📝 Query: {query}")
        print(f"🎯 Planner Output: {planner_output}")
        print(f"🔗 Flow: {' → '.join(planner_output)}")
        print("-" * 60)
        
        # Build dynamic graph
        workflow_graph = self.build_dynamic_graph(planner_output)
        
        # Compile the graph
        app = workflow_graph.compile()
        
        # Initialize state
        initial_state = AgentWorkflowState(
            query=query,
            planner_output=planner_output,
            current_step=0,
            agent_results={},
            accumulated_context="",
            final_answer="",
            workflow_complete=False,
            execution_path=[]
        )
        
        # Execute the workflow
        print(f"⚡ Executing Dynamic Workflow...")
        final_state = await app.ainvoke(initial_state)
        
        print(f"✅ Workflow Complete!")
        print(f"📊 Execution Path: {' → '.join(final_state['execution_path'])}")
        
        return {
            "query": query,
            "planner_output": planner_output,
            "execution_path": final_state["execution_path"],
            "agent_results": final_state["agent_results"],
            "final_answer": final_state["final_answer"],
            "workflow_complete": final_state["workflow_complete"],
            "total_steps": final_state["current_step"]
        }

class LangGraphAgentOrchestrator:
    """Main orchestrator using LangGraph for dynamic agent workflows"""
    
    def __init__(self, llm):
        self.flow_builder = DynamicAgentFlowLangGraph(llm)
    
    async def process_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple scenarios with different agent flows"""
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*80}")
            print(f"🎯 SCENARIO {i}")
            
            result = await self.flow_builder.execute_dynamic_workflow(
                query=scenario["query"],
                planner_output=scenario["planner_output"]
            )
            
            results.append(result)
            
            # Display scenario results
            self._display_scenario_results(scenario, result)
        
        return results
    
    def _display_scenario_results(self, scenario: Dict[str, Any], result: Dict[str, Any]):
        """Display results for each scenario"""
        
        print(f"\n📋 SCENARIO RESULTS:")
        print(f"Query: {scenario['query']}")
        print(f"Agents Used: {len(result['agent_results'])}")
        print(f"Execution Path: {' → '.join(result['execution_path'])}")
        
        if result['final_answer']:
            print(f"Final Answer: {result['final_answer'][:200]}...")
        
        print(f"Steps Completed: {result['total_steps']}")

# Specialized Graphs for Different Scenarios
class ScenarioSpecificGraphs:
    """Pre-built graphs for common scenarios"""
    
    def __init__(self, llm):
        self.llm = llm
        self.flow_builder = DynamicAgentFlowLangGraph(llm)
    
    def create_single_agent_graph(self, agent_name: str) -> StateGraph:
        """Optimized graph for single agent scenarios"""
        
        workflow = StateGraph(AgentWorkflowState)
        
        # Add single agent node
        if agent_name in self.flow_builder.agent_functions:
            workflow.add_node(agent_name, self.flow_builder.agent_functions[agent_name])
            workflow.set_entry_point(agent_name)
            workflow.add_edge(agent_name, END)
        
        return workflow
    
    def create_synthesis_graph(self, planner_output: List[str]) -> StateGraph:
        """Optimized graph for workflows ending with insights agent"""
        
        workflow = StateGraph(AgentWorkflowState)
        
        # Add all agents in sequence
        for agent_name in planner_output:
            if agent_name in self.flow_builder.agent_functions:
                workflow.add_node(agent_name, self.flow_builder.agent_functions[agent_name])
        
        # Set up linear flow
        workflow.set_entry_point(planner_output[0])
        for i in range(len(planner_output) - 1):
            workflow.add_edge(planner_output[i], planner_output[i + 1])
        
        workflow.add_edge(planner_output[-1], END)
        
        return workflow

# Test Implementation
async def test_langgraph_scenarios():
    """Test the three scenarios using LangGraph"""
    
    # Setup LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create orchestrator
    orchestrator = LangGraphAgentOrchestrator(llm)
    
    # Test scenarios - exactly as specified
    scenarios = [
        {
            "query": "What was the previous purchase?",
            "planner_output": ["agent_purchase", "agent_account", "agent_generate_insights"]
        },
        {
            "query": "Is product in stock?",
            "planner_output": ["agent_inventory"]
        },
        {
            "query": "Complete customer analysis", 
            "planner_output": ["agent_purchase", "agent_account", "agent_support", "agent_analytics", "agent_generate_insights"]
        }
    ]
    
    # Execute all scenarios
    results = await orchestrator.process_scenarios(scenarios)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"🎯 LANGGRAPH EXECUTION SUMMARY")
    for i, result in enumerate(results, 1):
        print(f"Scenario {i}: {len(result['execution_path'])} agents, {result['total_steps']} steps")
    
    return results

# Advanced Features Demo
async def demo_advanced_langgraph_features():
    """Demonstrate advanced LangGraph features"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    flow_builder = DynamicAgentFlowLangGraph(llm)
    
    # Example: Conditional routing based on query type
    def create_conditional_graph():
        """Graph with conditional routing"""
        
        workflow = StateGraph(AgentWorkflowState)
        
        # Add nodes
        workflow.add_node("agent_purchase", flow_builder.agent_functions["agent_purchase"])
        workflow.add_node("agent_inventory", flow_builder.agent_functions["agent_inventory"])
        workflow.add_node("agent_generate_insights", flow_builder.agent_functions["agent_generate_insights"])
        
        # Conditional router
        def route_based_on_query(state: AgentWorkflowState) -> str:
            query = state["query"].lower()
            if "stock" in query or "inventory" in query:
                return "agent_inventory"
            elif "purchase" in query:
                return "agent_purchase"
            else:
                return "agent_generate_insights"
        
        workflow.set_conditional_entry_point(route_based_on_query)
        workflow.add_edge("agent_purchase", "agent_generate_insights")
        workflow.add_edge("agent_inventory", END)
        workflow.add_edge("agent_generate_insights", END)
        
        return workflow
    
    # Test conditional routing
    conditional_app = create_conditional_graph().compile()
    
    test_queries = [
        "Is the product in stock?",
        "What was my last purchase?",
        "Give me general insights"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        initial_state = AgentWorkflowState(
            query=query,
            planner_output=[],
            current_step=0,
            agent_results={},
            accumulated_context="",
            final_answer="",
            workflow_complete=False,
            execution_path=[]
        )
        
        result = await conditional_app.ainvoke(initial_state)
        print(f"Routed to: {result['execution_path']}")

if __name__ == "__main__":
    # Run the main test
    asyncio.run(test_langgraph_scenarios())
    
    # Uncomment to run advanced features demo
    # asyncio.run(demo_advanced_langgraph_features())