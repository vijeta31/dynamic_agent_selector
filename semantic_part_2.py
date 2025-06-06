# dynamic_agent_flow_builder.py
import asyncio
from typing import List, Dict, Any, Optional
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.group_chat import GroupChat, GroupChatSettings
from semantic_kernel.agents.strategies import SequentialSelectionStrategy
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.functions import KernelArguments

class DynamicAgentFlowBuilder:
    """Builds dynamic agent flows from planner output using semantic_kernel.agents"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.agent_pool = self._initialize_agent_pool()
    
    def _initialize_agent_pool(self) -> Dict[str, ChatCompletionAgent]:
        """Initialize all available agents that planner can select from"""
        
        agents = {}
        
        # Purchase Agent - Specialized for purchase history analysis
        agents["agent_purchase"] = ChatCompletionAgent(
            service_id="purchase_service",
            kernel=self.kernel,
            name="PurchaseAgent",
            instructions="""
            You are a Purchase Analysis Agent specializing in customer purchase history.
            
            For queries about purchases:
            - Analyze transaction history and patterns
            - Identify previous purchases with details (products, amounts, dates)
            - Provide purchase insights and trends
            - Focus on specific purchase-related questions
            
            Always provide concrete purchase data when available.
            """,
            description="Analyzes customer purchase history and transaction patterns"
        )
        
        # Account Agent - Customer account management
        agents["agent_account"] = ChatCompletionAgent(
            service_id="account_service",
            kernel=self.kernel,
            name="AccountAgent", 
            instructions="""
            You are an Account Management Agent handling customer account details.
            
            Responsibilities:
            - Customer profile and membership status
            - Account health and payment history  
            - Account-related metrics and status
            - Customer loyalty and engagement levels
            
            Provide account context that supports other agents' analysis.
            """,
            description="Manages customer account information and membership details"
        )
        
        # Inventory Agent - Product availability and stock
        agents["agent_inventory"] = ChatCompletionAgent(
            service_id="inventory_service", 
            kernel=self.kernel,
            name="InventoryAgent",
            instructions="""
            You are an Inventory Management Agent handling stock and availability.
            
            Focus on:
            - Real-time stock levels and availability
            - Product inventory status
            - Stock alerts and availability forecasts
            - Inventory-related customer inquiries
            
            Provide immediate, actionable inventory information.
            """,
            description="Provides real-time inventory and product availability data"
        )
        
        # Support Agent - Customer service history
        agents["agent_support"] = ChatCompletionAgent(
            service_id="support_service",
            kernel=self.kernel, 
            name="SupportAgent",
            instructions="""
            You are a Customer Support Agent analyzing service interactions.
            
            Analyze:
            - Support ticket history and resolutions
            - Customer satisfaction ratings
            - Service interaction patterns
            - Support-related insights for customer health
            
            Contribute support context to overall customer analysis.
            """,
            description="Analyzes customer support history and satisfaction metrics"
        )
        
        # Analytics Agent - Customer behavior analytics  
        agents["agent_analytics"] = ChatCompletionAgent(
            service_id="analytics_service",
            kernel=self.kernel,
            name="AnalyticsAgent", 
            instructions="""
            You are a Customer Analytics Agent providing behavioral insights.
            
            Analyze:
            - Customer engagement patterns and metrics
            - Behavioral trends and lifecycle stage
            - Predictive customer insights
            - Cross-functional data correlation
            
            Generate data-driven insights about customer behavior.
            """,
            description="Provides customer analytics and behavioral pattern analysis"
        )
        
        # Insights Agent - Synthesis and strategic recommendations
        agents["agent_generate_insights"] = ChatCompletionAgent(
            service_id="insights_service",
            kernel=self.kernel,
            name="InsightsAgent",
            instructions="""
            You are a Strategic Insights Agent responsible for synthesis and recommendations.
            
            Your role:
            - Synthesize information from ALL previous agents in the workflow
            - Generate comprehensive insights combining multiple data sources  
            - Provide actionable recommendations based on complete analysis
            - Create strategic conclusions that address the original query
            
            IMPORTANT: Always reference and build upon previous agent outputs.
            Create cohesive insights that tie together all available information.
            """,
            description="Synthesizes multi-agent outputs into strategic insights and recommendations"
        )
        
        return agents
    
    async def build_dynamic_flow(self, query: str, planner_output: List[str]) -> Dict[str, Any]:
        """
        Main method - builds dynamic workflow from planner output
        This is where the magic happens!
        """
        
        print(f"🔧 BUILDING DYNAMIC AGENT FLOW")
        print(f"📝 Query: {query}")
        print(f"🎯 Planner Selected: {planner_output}")
        print(f"🔗 Flow: {' → '.join(planner_output)}")
        print("-" * 60)
        
        # Step 1: Validate and get selected agents
        selected_agents = self._get_selected_agents(planner_output)
        if not selected_agents:
            raise ValueError("No valid agents selected by planner")
        
        # Step 2: Determine optimal execution strategy based on agent combination
        execution_strategy = self._determine_execution_strategy(query, planner_output)
        
        # Step 3: Build and execute dynamic workflow
        workflow_result = await self._execute_dynamic_workflow(
            query=query,
            selected_agents=selected_agents, 
            strategy=execution_strategy,
            planner_output=planner_output
        )
        
        return workflow_result
    
    def _get_selected_agents(self, planner_output: List[str]) -> List[ChatCompletionAgent]:
        """Convert planner output to actual agent instances"""
        
        selected_agents = []
        for agent_name in planner_output:
            if agent_name in self.agent_pool:
                selected_agents.append(self.agent_pool[agent_name])
                print(f"✅ Added {agent_name} to workflow")
            else:
                print(f"⚠️  Agent {agent_name} not found in pool")
        
        return selected_agents
    
    def _determine_execution_strategy(self, query: str, planner_output: List[str]) -> str:
        """Determine best execution strategy based on scenario"""
        
        # Strategy logic based on agent combinations and query type
        if len(planner_output) == 1:
            return "single_agent"
        elif "agent_generate_insights" in planner_output:
            return "sequential_with_synthesis"  
        elif len(planner_output) <= 3:
            return "sequential_simple"
        else:
            return "sequential_complex"
    
    async def _execute_dynamic_workflow(self, 
                                      query: str,
                                      selected_agents: List[ChatCompletionAgent],
                                      strategy: str,
                                      planner_output: List[str]) -> Dict[str, Any]:
        """Execute the dynamic workflow based on strategy"""
        
        print(f"⚡ Executing Strategy: {strategy}")
        
        if strategy == "single_agent":
            return await self._execute_single_agent_flow(query, selected_agents[0])
        
        elif strategy == "sequential_with_synthesis":
            return await self._execute_sequential_synthesis_flow(query, selected_agents, planner_output)
        
        elif strategy in ["sequential_simple", "sequential_complex"]:
            return await self._execute_sequential_flow(query, selected_agents, planner_output)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _execute_single_agent_flow(self, query: str, agent: ChatCompletionAgent) -> Dict[str, Any]:
        """Handle single agent scenarios - like inventory check"""
        
        print(f"🎯 Single Agent Execution: {agent.name}")
        
        # Direct agent invocation
        message = ChatMessageContent(role=AuthorRole.USER, content=query)
        response = await agent.invoke(message)
        
        return {
            "execution_type": "single_agent",
            "agent_used": agent.name,
            "query": query,
            "result": str(response),
            "workflow_complete": True
        }
    
    async def _execute_sequential_synthesis_flow(self, 
                                               query: str, 
                                               selected_agents: List[ChatCompletionAgent],
                                               planner_output: List[str]) -> Dict[str, Any]:
        """Handle flows that end with insights agent - requires context passing"""
        
        print(f"🔄 Sequential Synthesis Flow with {len(selected_agents)} agents")
        
        # Setup group chat for sequential execution
        settings = GroupChatSettings(
            selection_strategy=SequentialSelectionStrategy(),
            max_turns=len(selected_agents),
            terminate_on_agent_leave=True
        )
        
        group_chat = GroupChat(agents=selected_agents, settings=settings)
        
        # Execute with context accumulation
        workflow_history = []
        context_summary = ""
        
        # Initial message
        initial_message = ChatMessageContent(
            role=AuthorRole.USER, 
            content=f"""
            Original Query: {query}
            
            This is a sequential workflow with {len(selected_agents)} agents.
            Each agent should build upon previous results.
            
            Current Context: [Starting workflow]
            """
        )
        
        # Execute sequential flow
        async for response in group_chat.invoke_stream(initial_message):
            workflow_step = {
                "agent": response.agent_name,
                "content": str(response.content),
                "step": len(workflow_history) + 1
            }
            workflow_history.append(workflow_step)
            
            # Accumulate context for next agent
            context_summary += f"\n{response.agent_name}: {str(response.content)[:200]}..."
            
            print(f"📋 Step {workflow_step['step']}: {response.agent_name}")
            print(f"   Content: {str(response.content)[:100]}...")
        
        return {
            "execution_type": "sequential_with_synthesis",
            "planner_output": planner_output,
            "query": query, 
            "workflow_history": workflow_history,
            "context_summary": context_summary,
            "final_result": workflow_history[-1] if workflow_history else None,
            "workflow_complete": True
        }
    
    async def _execute_sequential_flow(self, 
                                     query: str,
                                     selected_agents: List[ChatCompletionAgent], 
                                     planner_output: List[str]) -> Dict[str, Any]:
        """Handle standard sequential flows"""
        
        print(f"➡️  Standard Sequential Flow with {len(selected_agents)} agents")
        
        # Similar to synthesis flow but simpler context handling
        settings = GroupChatSettings(
            selection_strategy=SequentialSelectionStrategy(),
            max_turns=len(selected_agents),
            terminate_on_agent_leave=True
        )
        
        group_chat = GroupChat(agents=selected_agents, settings=settings)
        
        workflow_results = []
        initial_message = ChatMessageContent(role=AuthorRole.USER, content=query)
        
        async for response in group_chat.invoke_stream(initial_message):
            step_result = {
                "agent": response.agent_name,
                "response": str(response.content),
                "step_number": len(workflow_results) + 1
            }
            workflow_results.append(step_result)
            print(f"✅ {response.agent_name}: Completed")
        
        return {
            "execution_type": "sequential_standard",
            "planner_output": planner_output,
            "query": query,
            "workflow_results": workflow_results,
            "workflow_complete": True
        }

# Complete Integration with Test Scenarios
class DynamicAgentOrchestrator:
    """Main orchestrator that handles planner output and builds dynamic flows"""
    
    def __init__(self, kernel: Kernel):
        self.flow_builder = DynamicAgentFlowBuilder(kernel)
    
    async def process_planner_output(self, query: str, planner_output: List[str]) -> Dict[str, Any]:
        """Main entry point - processes planner output into dynamic agent flow"""
        
        print(f"\n{'='*80}")
        print(f"🚀 DYNAMIC AGENT ORCHESTRATION")
        
        # Build and execute dynamic flow
        result = await self.flow_builder.build_dynamic_flow(query, planner_output)
        
        print(f"✅ WORKFLOW COMPLETED")
        print(f"📊 Execution Type: {result.get('execution_type')}")
        
        return result

# Test the three scenarios
async def test_dynamic_scenarios():
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    
    # Setup
    kernel = Kernel()
    service = OpenAIChatCompletion(ai_model_id="gpt-4", api_key="your-key") 
    kernel.add_service(service)
    
    orchestrator = DynamicAgentOrchestrator(kernel)
    
    # Test scenarios from your example
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
    
    # Execute each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}")
        
        result = await orchestrator.process_planner_output(
            query=scenario["query"],
            planner_output=scenario["planner_output"]
        )
        
        # Display results based on execution type
        if result["execution_type"] == "single_agent":
            print(f"🎯 Single Agent Result: {result['result'][:150]}...")
            
        elif result["execution_type"] == "sequential_with_synthesis":
            print(f"🔄 Sequential Synthesis - {len(result['workflow_history'])} steps")
            if result["final_result"]:
                print(f"📝 Final Insight: {result['final_result']['content'][:150]}...")
                
        elif result["execution_type"] == "sequential_standard":
            print(f"➡️  Sequential Standard - {len(result['workflow_results'])} agents")
            for step in result['workflow_results']:
                print(f"   {step['agent']}: {step['response'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_dynamic_scenarios())