# langgraph_poc.py
from typing import List, Dict, Any, TypedDict
from langgraph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncio

class AgentState(TypedDict):
    query: str
    selected_agents: List[str]
    agent_results: Dict[str, str]
    final_answer: str

class MultiAgentSystem:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Define available agents
        self.available_agents = {
            "agent_purchase": "Retrieves customer purchase history and transaction details",
            "agent_account": "Gets customer account information and profile data", 
            "agent_inventory": "Checks product availability and inventory status",
            "agent_support": "Handles customer service and support queries",
            "agent_analytics": "Provides data analysis and business insights",
            "agent_generate_insights": "Synthesizes information to generate actionable insights"
        }
        
        # Build the graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Planner agent that selects which agents to invoke"""
        
        available_agents_str = "\n".join([
            f"- {name}: {desc}" for name, desc in self.available_agents.items()
        ])
        
        planning_prompt = f"""
        You are a planner agent. Given a user query, select which agents should be invoked to answer it.
        
        Available agents:
        {available_agents_str}
        
        User Query: {state['query']}
        
        Return only the agent names that should be invoked, separated by commas.
        Example: agent_purchase, agent_account, agent_generate_insights
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=planning_prompt),
            HumanMessage(content=state['query'])
        ])
        
        selected_agents = [agent.strip() for agent in response.content.split(',')]
        
        return {
            **state,
            "selected_agents": selected_agents
        }
    
    async def _executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute the selected agents"""
        
        agent_results = {}
        
        # Execute agents in parallel
        tasks = []
        for agent_name in state['selected_agents']:
            if agent_name in self.available_agents:
                tasks.append(self._execute_agent(agent_name, state['query']))
        
        results = await asyncio.gather(*tasks)
        
        for i, agent_name in enumerate(state['selected_agents']):
            if agent_name in self.available_agents:
                agent_results[agent_name] = results[i]
        
        return {
            **state,
            "agent_results": agent_results
        }
    
    async def _execute_agent(self, agent_name: str, query: str) -> str:
        """Simulate individual agent execution"""
        
        agent_prompts = {
            "agent_purchase": f"You are a purchase history agent. For the query '{query}', provide relevant purchase information. Return: 'Customer made a large purchase of DCC instruments worth $2400 last month.'",
            "agent_account": f"You are an account information agent. For the query '{query}', provide account details. Return: 'Customer is Amazon Prime member since 2019, excellent payment history.'",
            "agent_inventory": f"You are an inventory agent. For the query '{query}', check product availability. Return: 'DCC instruments currently in stock, 150 units available.'",
            "agent_support": f"You are a support agent. For the query '{query}', provide support context. Return: 'No recent support tickets, customer satisfaction rating: 4.8/5.'",
            "agent_analytics": f"You are an analytics agent. For the query '{query}', provide analytical insights. Return: 'Customer shows high engagement, 15% above average order value.'",
            "agent_generate_insights": f"You are an insights generation agent. For the query '{query}', provide strategic insights. Return: 'Customer demonstrates strong loyalty and high-value purchasing behavior.'"
        }
        
        prompt = agent_prompts.get(agent_name, f"Process query: {query}")
        
        response = await self.llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ])
        
        return response.content
    
    async def _synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize results from all agents"""
        
        results_summary = "\n".join([
            f"{agent}: {result}" for agent, result in state['agent_results'].items()
        ])
        
        synthesis_prompt = f"""
        Based on the following agent results, provide a comprehensive final answer to the user's query.
        
        Original Query: {state['query']}
        
        Agent Results:
        {results_summary}
        
        Provide a clear, concise final answer that incorporates the relevant information.
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=state['query'])
        ])
        
        return {
            **state,
            "final_answer": response.content
        }
    
    async def run(self, query: str) -> Dict[str, Any]:
        """Run the complete multi-agent workflow"""
        
        initial_state = AgentState(
            query=query,
            selected_agents=[],
            agent_results={},
            final_answer=""
        )
        
        result = await self.workflow.ainvoke(initial_state)
        return result

# Usage
async def main():
    system = MultiAgentSystem()
    
    query = "What was the previous purchase of the Amazon customer?"
    result = await system.run(query)
    
    print(f"Query: {result['query']}")
    print(f"Selected Agents: {result['selected_agents']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Agents Used: {list(result['agent_results'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())