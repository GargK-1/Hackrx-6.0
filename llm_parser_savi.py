# llm_parser.py
# -- coding: utf-8 --

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from enum import Enum
from typing import Optional, List
import asyncio, os, re
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# 1. ENUM – identical to your previous one                                    #
# --------------------------------------------------------------------------- #
class QueryType(str, Enum):
    yes_no         = "yes_no"
    definition     = "definition"
    numeric_factoid= "numeric_factoid"
    listing        = "listing"
    sub_limit      = "sub_limit"
    procedural     = "procedural"
    eligibility    = "eligibility"
    others         = "others"

# --------------------------------------------------------------------------- #
# 2. PROMPT – asks for *four* keys only                                       #
# --------------------------------------------------------------------------- #
_TEMPLATE = """
You are an assistant that converts a user’s insurance-policy question into a concise JSON object.
*Return **only** a JSON object – no extra text.*  
Use **snake_case** in every string.

###Your primary goal is to determine if the user's query contains one single topic or multiple distinct topics. Your JSON output format will adapt based on this determination.###

The JSON must have **exactly four keys**:

** Case 1: Single Topic Query **
1. **key_word**  – the single most important phrase (1-4 words) the retrieval engine should weight higher.
2. **sub_query** – a list of short, standalone questions that together cover every aspect of the user’s query.  
   • If the user is really asking just one thing, put one concise sentence in the list.  
3. **raw_query** – the original user query (verbatim).  
4. **query_type** – choose one from: yes_no, definition, numeric_factoid, listing, sub_limit, procedural, eligibility, others.

** Case 2: Multiple Distinct Topics **
1. **key_word**  –  Identify the most important key phrase for each distinct topic.  Identify the most important key phrase for each distinct topic. Combine these into a single string, separated by a semicolon and a space  
2. **sub_query** – a list of short, standalone questions that cover each distinct topic in the user's query. 
    • If the user is asking about multiple topics, ensure each topic has its own single question in the list.
    • Mapping should be one-to-one: first sub_query should correspond to first key_word, second sub_query to second key_word, and so on.
3. **raw_query** – the original user query (verbatim).  
4. **query_type** – for multiple topics, it will be others.

### Examples

User: "Does this policy cover maternity expenses, and what are the conditions?"  
JSON:
{{
  "key_word": "maternity_expenses",
  "sub_query": ["coverage_and_conditions_of_maternity_expenses"],
  "raw_query": "Does this policy cover maternity expenses, and what are the conditions?",
  "query_type": "yes_no"
}}

User: "For a claim submission involving robotic surgery for a spouse at "Apollo Care Hospital" (city not specified), what supporting documents are needed, how to confirm if the hospital is a network provider, and can a sibling above 26 continue as a dependent if financially dependent after job loss?"  
JSON:
{{
  "key_word": "claim_submission_robotic_surgery; sibling_dependent",
  "sub_query": ["what_documents_are_required_for_a_robotic_surgery_claim", "how_to_confirm_if_a_hospital_is_a_network_provider", "what_is_the_eligibility_for_a_financially_dependent_sibling_over_26"],
  "raw_query": "For a claim submission involving robotic surgery for a spouse at "Apollo Care Hospital" (city not specified), what supporting documents are needed, how to confirm if the hospital is a network provider, and can a sibling above 26 continue as a dependent if financially dependent after job loss?",
  "query_type": "others"
}}

User: "How does the policy define a hospital?"  
JSON:
{{
  "key_word": "hospital_definition",
  "sub_query": ["definition_of_hospital"],
  "raw_query": "How does the policy define a hospital?",
  "query_type": "definition"
}}

User: "While inquiring about the maximum cashless hospitalization benefit for accidental trauma for a covered parent-in-law, simultaneously provide the claim notification procedure, and confirm the process to replace a lost ID card for another dependent."  
JSON:
{{
  "key_word": "accidental_trauma_benefit; claim_notification_procedure; lost_id_card_replacement",
  "sub_query": ["what_is_the_maximum_cashless_benefit_for_accidental_trauma","what_is_the_procedure_for_notifying_a_claim","what_is_the_process_to_replace_a_lost_id_card"],
  "raw_query": "While inquiring about the maximum cashless hospitalization benefit for accidental trauma for a covered parent-in-law, simultaneously provide the claim notification procedure, and confirm the process to replace a lost ID card for another dependent.",
  "query_type": "others"
}}

User: "What is the waiting period for pre-existing diseases to be covered?"  
JSON:
{{
  "key_word": "pre_existing_diseases",
  "sub_query": ["waiting_period_for_pre_existing_diseases"],
  "raw_query": "What is the waiting period for pre-existing diseases to be covered?",
  "query_type": "numeric_factoid"
}}

User: "What documents must I submit to file a hospitalization claim?"  
JSON:
{{
  "key_word": "hospitalization_claim_documents",
  "sub_query": ["required_documents_for_hospitalization_claim"],
  "raw_query": "What documents must I submit to file a hospitalization claim?",
  "query_type": "listing"
}}

User: "What is the daily room-rent sub-limit under Plan A?"  
JSON:
{{
  "key_word": "room_rent_sub_limit",
  "sub_query": ["daily_room_rent_sub_limit_under_plan_a"],
  "raw_query": "What is the daily room rent sub-limit under Plan A?",
  "query_type": "sub_limit"
}}

User: "How many days will it take the insurer to settle my claim after submission?"  
JSON:
{{
  "key_word": "claim_settlement_timeline",
  "sub_query": ["settlement_period_after_claim_submission"],
  "raw_query": "How many days will it take the insurer to settle my claim after submission?",
  "query_type": "procedural"
}}

User: "Who is eligible for the preventive health check-up benefit?"  
JSON:
{{
  "key_word": "preventive_health_checkup",
  "sub_query": ["eligibility_for_preventive_health_checkup_benefit"],
  "raw_query": "Who is eligible for the preventive health check-up benefit?",
  "query_type": "eligibility"
}}

User: "Describe the steps to port a prior individual policy from another insurer for a dependent parent-in-law, list documents needed for a post-hospitalization medicine claim for your child, and provide the toll-free customer service number."  
JSON:
{{
  "key_word": "policy_portability_steps; post_hospitalization_claim_documents; customer_service_number",
  "sub_query": ["what_are_the_steps_to_port_a_health_insurance_policy", "what_documents_are_needed_for_a_post_hospitalization_claim", "what_is_the_toll_free_customer_service_number"],
  "raw_query": "Describe the steps to port a prior individual policy from another insurer for a dependent parent-in-law, list documents needed for a post-hospitalization medicine claim for your child, and provide the toll-free customer service number.",
  "query_type": "others"
}}

User: "Compare the benefits of Plan A versus Plan B under this policy."  
JSON:
{{
  "key_word": "plan_comparison",
  "sub_query": ["benefit_differences_between_plan_a_and_plan_b"],
  "raw_query": "Compare the benefits of Plan A versus Plan B under this policy.",
  "query_type": "others"
}}

User: "What is the no-claim discount (NCD) offered in this policy?"  
JSON:
{{
  "key_word": "no_claim_discount",
  "sub_query": ["percentage_of_no_claim_discount_offered"],
  "raw_query": "What is the no Claim Discount (NCD) offered in this policy?",
  "query_type": "numeric_factoid"
}}

User: "Is cataract surgery covered, and what is the waiting period?"  
JSON:
{{
  "key_word": "cataract_surgery",
  "sub_query": ["coverage_of_cataract_surgery", "waiting_period_for_cataract_surgery"],
  "raw_query": "Is cataract surgery covered, and what is the waiting period?",
  "query_type": "yes_no"
}}


Now convert the following user query.

User: "{query}"
JSON:
"""

# --------------------------------------------------------------------------- #
# 3. Pydantic schema matching the four keys                                   #
# --------------------------------------------------------------------------- #
class StructuredQuery(BaseModel):
    key_word: str = Field(..., description="Main phrase to boost in retrieval")
    sub_query: List[str] = Field(..., description="List of standalone sub-questions")
    raw_query: str = Field(..., description="Original user query")
    query_type: QueryType = Field(..., description="High-level category")

# --------------------------------------------------------------------------- #
# 4. Parser class                                                             #
# --------------------------------------------------------------------------- #
class QueryParser:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=StructuredQuery)

        self.output_fixing_parser = OutputFixingParser.from_llm(
            parser=self.parser, llm=self.llm
        )

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=_TEMPLATE
        )
        self.chain = self.prompt | self.llm | self.output_fixing_parser

    async def parse_query(self, user_query: str) -> StructuredQuery:
        print(f"\nParsing query: '{user_query}'")
        result: StructuredQuery = await self.chain.ainvoke({"query": user_query})

        # optional: normalise key_word for easier downstream matching
        result.key_word = re.sub(r"[^a-z0-9]+", " ", result.key_word.lower()).strip()

        return result

# --------------------------------------------------------------------------- #
# 5. Quick CLI test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")

    qp = QueryParser(llm)
    q = "When will my root canal claim of Rs 25,000 be settled?"
    parsed = asyncio.run(qp.parse_query(q))
    print(parsed.model_dump_json(indent=2))




# """
# You are an assistant that converts a user’s insurance-policy question into a concise JSON object.
# *Return **only** a JSON object – no extra text.*  
# Use **snake_case** in every string.

# The JSON must have **exactly four keys**:

# 1. **key_word**  – the single most important phrase (1-4 words) the retrieval engine should weight higher.
# 2. **sub_query** – a list of short, standalone questions that together cover every aspect of the user’s query.  
#    • If the user is really asking just one thing, put one concise sentence in the list.  
# 3. **raw_query** – the original user query (verbatim).  
# 4. **query_type** – choose one from: yes_no, definition, numeric_factoid, listing, sub_limit, procedural, eligibility, others.

# ### Examples

# User: "Does this policy cover maternity expenses, and what are the conditions?"  
# JSON:
# {{
#   "key_word": "maternity_expenses",
#   "sub_query": ["coverage_of_maternity_expenses", "conditions_for_maternity_coverage"],
#   "raw_query": "Does this policy cover maternity expenses, and what are the conditions?",
#   "query_type": "yes_no"
# }}

# User: "How does the policy define a hospital?"  
# JSON:
# {{
#   "key_word": "hospital_definition",
#   "sub_query": ["definition_of_hospital"],
#   "raw_query": "How does the policy define a hospital?",
#   "query_type": "definition"
# }}

# User: "What is the waiting period for pre-existing diseases to be covered?"  
# JSON:
# {{
#   "key_word": "pre_existing_diseases",
#   "sub_query": ["waiting_period_for_pre_existing_diseases"],
#   "raw_query": "What is the waiting period for pre-existing diseases to be covered?",
#   "query_type": "numeric_factoid"
# }}

# User: "What documents must I submit to file a hospitalization claim?"  
# JSON:
# {{
#   "key_word": "hospitalization_claim_documents",
#   "sub_query": ["required_documents_for_hospitalization_claim"],
#   "raw_query": "What documents must I submit to file a hospitalization claim?",
#   "query_type": "listing"
# }}

# User: "What is the daily room-rent sub-limit under Plan A?"  
# JSON:
# {{
#   "key_word": "room_rent_sub_limit",
#   "sub_query": ["daily_room_rent_sub_limit_under_plan_a"],
#   "raw_query": "What is the daily room rent sub-limit under Plan A?",
#   "query_type": "sub_limit"
# }}

# User: "How many days will it take the insurer to settle my claim after submission?"  
# JSON:
# {{
#   "key_word": "claim_settlement_timeline",
#   "sub_query": ["settlement_period_after_claim_submission"],
#   "raw_query": "How many days will it take the insurer to settle my claim after submission?",
#   "query_type": "procedural"
# }}

# User: "Who is eligible for the preventive health check-up benefit?"  
# JSON:
# {{
#   "key_word": "preventive_health_checkup",
#   "sub_query": ["eligibility_for_preventive_health_checkup_benefit"],
#   "raw_query": "Who is eligible for the preventive health check-up benefit?",
#   "query_type": "eligibility"
# }}

# User: "Compare the benefits of Plan A versus Plan B under this policy."  
# JSON:
# {{
#   "key_word": "plan_comparison",
#   "sub_query": ["benefit_differences_between_plan_a_and_plan_b"],
#   "raw_query": "Compare the benefits of Plan A versus Plan B under this policy.",
#   "query_type": "others"
# }}

# User: "What is the no-claim discount (NCD) offered in this policy?"  
# JSON:
# {{
#   "key_word": "no_claim_discount",
#   "sub_query": ["percentage_of_no_claim_discount_offered"],
#   "raw_query": "What is the no Claim Discount (NCD) offered in this policy?",
#   "query_type": "numeric_factoid"
# }}

# User: "Is cataract surgery covered, and what is the waiting period?"  
# JSON:
# {{
#   "key_word": "cataract_surgery",
#   "sub_query": ["coverage_of_cataract_surgery", "waiting_period_for_cataract_surgery"],
#   "raw_query": "Is cataract surgery covered, and what is the waiting period?",
#   "query_type": "yes_no"
# }}


# Now convert the following user query.

# User: "{query}"
# JSON:
# """



























# """
# You are an assistant that converts a user’s insurance-policy question into a concise JSON object.
# *Return **only** a JSON object – no extra text.*  
# Use **snake_case** in every string.

# ###Your primary goal is to determine if the user's query contains one single topic or multiple distinct topics. Your JSON output format will adapt based on this determination.###

# The JSON must have **exactly four keys**:

# ** Case 1: Single Topic Query **
# 1. **key_word**  – the single most important phrase (1-4 words) the retrieval engine should weight higher.
# 2. **sub_query** – a list of short, standalone questions that together cover every aspect of the user’s query.  
#    • If the user is really asking just one thing, put one concise sentence in the list.  
# 3. **raw_query** – the original user query (verbatim).  
# 4. **query_type** – choose one from: yes_no, definition, numeric_factoid, listing, sub_limit, procedural, eligibility, others.

# ** Case 2: Multiple Distinct Topics **
# 1. **key_word**  –  Identify the most important key phrase for each distinct topic.  Identify the most important key phrase for each distinct topic. Combine these into a single string, separated by a semicolon and a space  
# 2. **sub_query** – a list of short, standalone questions that cover each distinct topic in the user's query. 
#     • If the user is asking about multiple topics, ensure each topic has its own single question in the list.
#     • Mapping should be one-to-one: first sub_query should correspond to first key_word, second sub_query to second key_word, and so on.
# 3. **raw_query** – the original user query (verbatim).  
# 4. **query_type** – for multiple topics, it will be others.

# ### Examples

# User: "Does this policy cover maternity expenses, and what are the conditions?"  
# JSON:
# {{
#   "key_word": "maternity_expenses",
#   "sub_query": ["coverage_and_conditions_of_maternity_expenses"],
#   "raw_query": "Does this policy cover maternity expenses, and what are the conditions?",
#   "query_type": "yes_no"
# }}

# User: "For a claim submission involving robotic surgery for a spouse at "Apollo Care Hospital" (city not specified), what supporting documents are needed, how to confirm if the hospital is a network provider, and can a sibling above 26 continue as a dependent if financially dependent after job loss?"  
# JSON:
# {{
#   "key_word": "claim_submission_robotic_surgery; sibling_dependent",
#   "sub_query": ["what_documents_are_required_for_a_robotic_surgery_claim", "how_to_confirm_if_a_hospital_is_a_network_provider", "what_is_the_eligibility_for_a_financially_dependent_sibling_over_26"],
#   "raw_query": "For a claim submission involving robotic surgery for a spouse at "Apollo Care Hospital" (city not specified), what supporting documents are needed, how to confirm if the hospital is a network provider, and can a sibling above 26 continue as a dependent if financially dependent after job loss?",
#   "query_type": "others"
# }}

# User: "How does the policy define a hospital?"  
# JSON:
# {{
#   "key_word": "hospital_definition",
#   "sub_query": ["definition_of_hospital"],
#   "raw_query": "How does the policy define a hospital?",
#   "query_type": "definition"
# }}

# User: "While inquiring about the maximum cashless hospitalization benefit for accidental trauma for a covered parent-in-law, simultaneously provide the claim notification procedure, and confirm the process to replace a lost ID card for another dependent."  
# JSON:
# {{
#   "key_word": "accidental_trauma_benefit; claim_notification_procedure; lost_id_card_replacement",
#   "sub_query": ["what_is_the_maximum_cashless_benefit_for_accidental_trauma","what_is_the_procedure_for_notifying_a_claim","what_is_the_process_to_replace_a_lost_id_card"],
#   "raw_query": "While inquiring about the maximum cashless hospitalization benefit for accidental trauma for a covered parent-in-law, simultaneously provide the claim notification procedure, and confirm the process to replace a lost ID card for another dependent.",
#   "query_type": "others"
# }}

# User: "What is the waiting period for pre-existing diseases to be covered?"  
# JSON:
# {{
#   "key_word": "pre_existing_diseases",
#   "sub_query": ["waiting_period_for_pre_existing_diseases"],
#   "raw_query": "What is the waiting period for pre-existing diseases to be covered?",
#   "query_type": "numeric_factoid"
# }}

# User: "What documents must I submit to file a hospitalization claim?"  
# JSON:
# {{
#   "key_word": "hospitalization_claim_documents",
#   "sub_query": ["required_documents_for_hospitalization_claim"],
#   "raw_query": "What documents must I submit to file a hospitalization claim?",
#   "query_type": "listing"
# }}

# User: "What is the daily room-rent sub-limit under Plan A?"  
# JSON:
# {{
#   "key_word": "room_rent_sub_limit",
#   "sub_query": ["daily_room_rent_sub_limit_under_plan_a"],
#   "raw_query": "What is the daily room rent sub-limit under Plan A?",
#   "query_type": "sub_limit"
# }}

# User: "How many days will it take the insurer to settle my claim after submission?"  
# JSON:
# {{
#   "key_word": "claim_settlement_timeline",
#   "sub_query": ["settlement_period_after_claim_submission"],
#   "raw_query": "How many days will it take the insurer to settle my claim after submission?",
#   "query_type": "procedural"
# }}

# User: "Who is eligible for the preventive health check-up benefit?"  
# JSON:
# {{
#   "key_word": "preventive_health_checkup",
#   "sub_query": ["eligibility_for_preventive_health_checkup_benefit"],
#   "raw_query": "Who is eligible for the preventive health check-up benefit?",
#   "query_type": "eligibility"
# }}

# User: "Describe the steps to port a prior individual policy from another insurer for a dependent parent-in-law, list documents needed for a post-hospitalization medicine claim for your child, and provide the toll-free customer service number."  
# JSON:
# {{
#   "key_word": "policy_portability_steps; post_hospitalization_claim_documents; customer_service_number",
#   "sub_query": ["what_are_the_steps_to_port_a_health_insurance_policy", "what_documents_are_needed_for_a_post_hospitalization_claim", "what_is_the_toll_free_customer_service_number"],
#   "raw_query": "Describe the steps to port a prior individual policy from another insurer for a dependent parent-in-law, list documents needed for a post-hospitalization medicine claim for your child, and provide the toll-free customer service number.",
#   "query_type": "others"
# }}

# User: "Compare the benefits of Plan A versus Plan B under this policy."  
# JSON:
# {{
#   "key_word": "plan_comparison",
#   "sub_query": ["benefit_differences_between_plan_a_and_plan_b"],
#   "raw_query": "Compare the benefits of Plan A versus Plan B under this policy.",
#   "query_type": "others"
# }}

# User: "What is the no-claim discount (NCD) offered in this policy?"  
# JSON:
# {{
#   "key_word": "no_claim_discount",
#   "sub_query": ["percentage_of_no_claim_discount_offered"],
#   "raw_query": "What is the no Claim Discount (NCD) offered in this policy?",
#   "query_type": "numeric_factoid"
# }}

# User: "Is cataract surgery covered, and what is the waiting period?"  
# JSON:
# {{
#   "key_word": "cataract_surgery",
#   "sub_query": ["coverage_of_cataract_surgery", "waiting_period_for_cataract_surgery"],
#   "raw_query": "Is cataract surgery covered, and what is the waiting period?",
#   "query_type": "yes_no"
# }}


# Now convert the following user query.

# User: "{query}"
# JSON:
# """