def run_tests_trump(qa):
    print(qa.invoke({"question": "Which country runs the world's greatest trading surplus, and which country runs the greatest deficit?"})['answer'])
    print(qa.invoke({"question": "Which country runs the world's greatest trading surplus, and which country runs the greatest deficit?"})['source_documents'])

def run_tests_company(qa):
    sys_prompt = '''System prompt: You are a highly experienced supply chain consultant with expertise in global logistics, procurement strategies, risk management, and cost optimization. You specialize in analyzing complex data from diverse sources, including market trends, weather reports, political developments, and global supply chain patterns, to provide actionable insights for improving supply chain efficiency. Your advice is practical, detail-oriented, and tailored to businesses aiming to lower costs, manage risks, and maximize profitability while ensuring sustainability and resilience. '''
            
    prompt = '''Prompt: Analyze the provided documents, which include news articles, charts/graphs, weather reports, political climate data, and global supply chain patterns. Based on this information, recommend actionable strategies to optimize my business supply chain efficiency. Focus on lowering costs, managing risks, and increasing profits. Your recommendations should address the following:
                1	Identifying potential supply chain bottlenecks or risks due to external factors (e.g., weather disruptions, geopolitical tensions).
                2	Suggesting cost-effective sourcing or logistics alternatives to improve profitability.
                3	Leveraging trends in global supply chain patterns to create competitive advantages.
                4	Incorporating strategies for sustainable and resilient supply chain management.
                Present your insights in a clear, actionable format, highlighting specific steps my business can take to enhance overall efficiency and adaptability.'''
    
    business_info = {
        "BasicCompanyInfo": {
            "companyName": "",
            "industry": "electronics",
            "headquartersLocation": "",
            "annualRevenueRange": "",
            "numberOfEmployees": ""
        },
        "Sustainability": {
            "sustainabilityGoals": "",
            "environmentalImpactMetrics": "",
            "ethicalSourcingPractices": ""
        },
        "SupplyChainOverview": {
            "primarySourcingLocations": "Taiwan, Ukraine, China",
            "criticalSuppliers": "Fabricates semi conductor chips in Taiwan\nNeon Supply in Ukraine\nImports microchips from China",
            "transportationModes": "",
            "keyTransportationHubs": "",
            "warehouseLocations": ""
        },
        "RiskAndResilience": {
            "majorRiskFactors": "",
            "recentSupplyChainDisruptions": ""
        },
        "ProductDetails": {
            "criticalMaterials": "",
            "environmentallyFriendlyAlternatives": ""
        },
        "ComplianceAndStandards": {
            "keyComplianceRequirements": "",
            "certificationsHeld": ""
        }
    }
    
    b = str(business_info)
    
    query = sys_prompt + prompt + b
    
    return query
    # print(qa.invoke({"question": query})['answer'])
    # print(qa.invoke({"question": query})['source_documents'])