import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# load_dotenv()


def get_embedding(text, model="text-embedding-3-large"):
    truncated_text = truncate_text_tokens(text)
    response = client.embeddings.create(input=[truncated_text], model=model).data
    response = response[0].embedding
    return response

def truncate_text_tokens(text):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(text)[:8191]

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

OPENAI_API_KEY = "" # put the OpenAI API Key 
client = OpenAI(api_key=OPENAI_API_KEY)

instruction = "Given the information from the Amlpytics, retrieve RFP (Request for Proposal) that best matches the information. Information: "
summary = """
Amplytics, a consulting firm specializing in data -driven decision -making, offers cutting -edge solutions tailored to the utilities industry. With a mission to transform utilities businesses into data -driven powerhouses, Amplytics integrates robust analytics , strategic advisory, and technological innovation to deliver actionable insights and measurable outcomes. Our offerings encompass end -to-end data solutions, predictive analytics, AI -driven modeling, digital transformation, and portfolio management, ensuri ng clients maximize the potential of their data for sustainable growth.  
Key Capabilities (Detailed Description)  
1. Data & Analytics  
Amplytics provides end -to-end data solutions that transform raw data into actionable insights 
through advanced analytics tools and methodologies, supporting streamlined executive decision -
making. In terms of data visualization and reporting, Amplytics util izes 1 -3-10 Lean Reporting 
frameworks to deliver concise, hierarchical data presentations. Custom dashboards designed using 
Power BI and Tableau ensure clear visual representations of critical data. Financial reporting services 
include variance analysis, b udget forecasting, portfolio financials, and program status updates, while 
field -to-executive reporting encompasses compliance dashboards, operational reviews, and data 
quality assessments.  
The analytics services offered by Amplytics are comprehensive and span multiple categories. 
Descriptive analytics focuses on summarizing historical data and uncovering trends through data 
mining and aggregation. Diagnostic analytics dives deeper, identifyi ng root causes of trends through 
data discovery and drill -down techniques. Predictive analytics employs sophisticated models to 
forecast future outcomes based on historical data patterns. Prescriptive analytics takes this a step 
further by implementing opti mization and simulation techniques to recommend actionable 
strategies and guide decision -making processes.  
2. Data Engineering & Data Science  
Amplytics specializes in data collection and ingestion, designing and managing robust data pipelines 
and ETL scripts that integrate data from diverse sources, whether structured or unstructured. For 
data storage and management, Amplytics implements secure,  scalable storage solutions on platforms 
such as SQL Server, Amazon S3, Azure, and Google Cloud, ensuring compliance with data governance 
standards and seamless retrieval.  
Predictive modeling and machine learning services allow Amplytics to identify patterns, trends, and 
future outcomes by building sophisticated models on platforms like Palantir Foundry. Machine 
learning algorithms are deployed to support advanced forecastin g and decision -making. Amplytics 
also leverages natural language processing (NLP) technologies to enhance customer engagement and 
understanding. Through the use of chatbots, sentiment analysis, and text classification, they enable 
organizations to gain dee per insights into customer preferences and feedback.  
3. Strategy & Advisory  
Amplytics offers a comprehensive portfolio strategy, managing project portfolios from planning 
through execution and incorporating risk management and performance evaluation at every stage. 
The firm provides strategic planning and execution services that a lign resources and timelines with 
organizational goals. Amplytics helps organizations define their visions and long -term objectives 
while fostering collaboration among stakeholders. To improve agility and decision -making, 
organizational structures are opti mized, and strategic performance monitoring systems are 
implemented to ensure key metrics are effectively evaluated.  Process improvement services are another core capability, where Amplytics introduces Lean 
Management initiatives to eliminate inefficiencies using value stream mapping. Six Sigma 
methodologies, such as DMAIC, are applied to enhance process consistency and quality. Business 
process reengineering focuses on improving cost, quality, and speed, while workflow automation 
reduces bottlenecks and increases productivity.  
In terms of portfolio and risk management, Amplytics offers Project Management Office (PMO) 
support to streamline project execution. Comprehensive risk assessments and mitigation strategies 
are conducted, including risk scoring, development of risk registe rs, and creation of contingency 
plans. The firm also provides compliance assessments and strategic advisory to help organizations 
navigate complex regulatory environments.  
4. Technology Innovation & AI  
Amplytics excels in software implementation by conducting thorough needs analyses to select 
solutions tailored to organizational goals. The firm manages the full lifecycle of software adoption, 
including configuration, customization, and integration into e xisting systems. Comprehensive user 
training programs and post -implementation support ensure smooth adoption and efficient 
utilization.  
Digital transformation is another area of expertise. Amplytics develops cohesive digital strategies 
that incorporate robotic process automation (RPA) and artificial intelligence tools, enhancing 
operational efficiency. Customer journeys are mapped to optim ize user experiences, and digital 
transformation initiatives are managed with a focus on aligning workstreams with strategic 
objectives.  
Generative AI capabilities at Amplytics include identifying use cases, assessing feasibility, and 
determining potential impacts. The firm develops and integrates custom AI models into workflows to 
drive operational efficiency and innovation. These models a re trained and fine -tuned to meet specific 
business requirements, ensuring optimal performance and alignment with organizational goals.  
This detailed description of Amplytics’ key capabilities highlights its commitment to providing 
comprehensive and tailored solutions that address the diverse challenges faced by organizations in 
the utilities industry and beyond.  
Proposed Solution  
Amplytics addresses the RFP requirements by beginning with a thorough needs analysis to ensure all 
proposed solutions align seamlessly with the client’s goals and objectives. This is followed by 
deploying a tailored strategy designed to incorporate advance d data analytics for precise decision -
making, machine learning models that anticipate future trends, and process automation to improve 
operational efficiency. The strategic advisory component is employed to guide clients through 
organizational challenges e ffectively. Additionally, Amplytics leverages its generative AI capabilities to 
identify high -impact use cases and deliver transformative, innovative solutions that meet specific 
client needs.  
Value Proposition  
Amplytics is committed to ensuring strategic alignment in all initiatives, guaranteeing that proposed 
solutions maximize the return on investment by aligning directly with organizational objectives. The 
company employs state -of-the-art tools and methodolog ies to ensure technological excellence in 
every aspect of its offerings. Operational efficiency is achieved by streamlining processes and 
addressing bottlenecks, creating a more agile and responsive operational framework. Finally, the 
scalability of Amplyti cs’ solutions ensures that implementations are flexible and able to grow 
alongside the client’s evolving business needs.  Deliverables  
Amplytics provides comprehensive dashboards that enable real -time insights into critical data, 
enhancing visibility and decision -making. Predictive models are developed to forecast trends 
accurately, empowering clients to anticipate changes and act proacti vely. Optimization roadmaps are 
delivered to enhance operational workflows and ensure processes are both effective and efficient. 
AI-powered solutions are tailored to address client -specific challenges, integrating seamlessly into 
existing operations to en hance performance and outcomes. These deliverables collectively provide a 
robust foundation for achieving the client’s strategic objectives.  
Conclusion  
Amplytics combines unparalleled expertise in data analytics, strategic advisory, and technology 
implementation to empower utilities businesses. By addressing every facet of the RFP with precision, 
Amplytics positions itself as the ideal partner to lead tra nsformative initiatives, ensuring long -term 
success and competitive advantage for the client.
"""

text_1 = """

 
 1 
Sources Sought Notice : SS-2022 -1007  
American -Made Utility Digital Transformation Prize  
 
The U.S. Department of Energy’s Office of Electricity in partnership with the National Renewable Energy 
Laboratory (NREL) are intending to launch the American -Made Utility Digital Transformation Prize in 
September 2022.  
This prize will be part of the American -Made Challenges program, which is your fast track to the clean 
energy revolution. Funded by the U.S. Department of Energy, we incentivize innovati on through prizes, 
training, teaming, and mentoring, connecting the nation’s entrepreneurs and innovators to America’s 
national labs and the private sector.  This prize is, in part, in response to the Electricity Advisory 
Committee’s report “ Big Data Analytics: Recommendations for the U.S. Department of Energy ” which 
outlined the need for DOE support in advancing data analytics for existing data sources within utilities.  
To inform the planning phase of the prize, NREL wishes to seek feedback on the prize structure and seek 
interest from utilities to participate.  
NREL is seeking feedback on two topics. We invite respondents to comment on one or both parts:  
• Part 1: Prize Structure –  Up to 1 page response  
• Part 2: Request for Uti lity Partners – Up to 2 page response  
 
This prize aims to connect utilities with interdisciplinary teams of software developers and data experts to 
facilitate transforming digital systems in the energy sector and data analytics for utilities. Utilities will 
provide details about interesting problems that they need to solve as well as access to related data. The 
goal of this prize is to have the teams work directly with utilities on their data, to solve their problems. 
Data could be energy use data, synchrop hasor data, weather data, fire assessment data, and more. The 
data will ideally be actual data but could also be synthetic data where applicable. Teams of developers 
will be paired with utilities to help tackle problems and build solutions specific to the utility to help the 
utility with their digital transformation needs.  
Utilizing a phased -prize approach, the prize will incentivize teams for coming to the table with a strong 
team and plan for execution. Throughout the prize, NREL and its partners will provide opportunities to 
help build teams. As the teams advance in the prize competition, they will work directly with utilities to 
help solve their problems and help develop software and data solutions that can be integral in the digital 
transformation strategy. Prize awards will be given to the top teams in each phase based on the work 
completed by the teams.  
Part 1: Prize Structure  
As a result of this RFI and prior to the prize launch, 3 -4 utility partners will identify issues or problems that 
they are f acing and present a plan for how they could utilize a software development team to overcome 
these issues or problems. Based on the identified problems, DOE will provide sample open -source 
datasets for competitor teams to utilize in their phase 1 submission s. 
Prize competitors will view the issues or problems and select which utility’s problem and the type of data 
they would like to work with. Teams will also propose ideas on how they could work with the data to 
provide actionable information and solve probl ems for the utilities.  AMERICAN -MADE | U.S. DEPARTMENT OF ENERGY   
 2 
In Phase 1 of the prize, teams of developers will form and propose a plan on how to address the issue. 
Partner utilities will join DOE to select finalists for each issue. Finalists will win a cash prize and be 
matched with a utility for the second phase of the prize.  
In Phase 2, utilities will mentor and work with teams as they develop their solutions. In Phase 2, it is 
expected that utilities would directly interact with and provide data to the 2 -4 teams selected in Phase 1 
to conti nue in Phase 2. At the end of phase 2, teams will present their solutions. 1 winner from each 
utility track will be selected as a winner, and a grand prize winner will be selected. Winners will receive 
cash prizes.  
NREL is seeking feedback on the prize str ucture. In 1 page or less, provide any feedback on the general 
outline, items that should be considered in the final prize rules, or suggestions for evaluation criteria.  
Part 2: Request for Utility Partners  
NREL is seeking 3 -4 utilities to engage as a part  of the prize. Utility partners would need to commit the 
following:  
• Provide a well- defined problem or issue they are facing,  
• Provide some criteria for evaluating team eligibility and experience,  
• Present a draft plan for how they would utilize a software development team , 
o Provide an example data set and/or short description of data types for phase 1; and  
o Provide a plan on how, in phase 2, the development team would have access to relevant 
data and/or access to relevant systems/platforms  for the duration of the prize. This 
could include, but is not limited to, utilizing an NDA or collaborative agreement, the utility 
creating open data sets for the team, and/or some other method,  
• Support the teams who are working on solutions for the utilities through mentorsh ip or 
consultation, serving as the “user” for the team’s solution(s),  
• Provide updates to DOE/NREL as to how the teams are performing, 
• Develop success story information that can be used to help promote the prize and provide impact 
metrics, and  
• Attend the N orth American Synchrophasor Initiative (NASPI) conferences for the duration of the 
prize and participate in relevant programing related to the prize.  
 
The utility partners will get the following benefits through this prize:  
• Gain access to developers willin g and able to help them solve tough technical issues related to 
digital transformation, specific to their system and its unique needs,  
• Potentially get several teams working on their issues, which may provide several options to viable 
solutions, and  
• DOE recognition for being a utility leading the charge in utility digital transformation through being 
a partner, mentor, and supporter.  
 
Utilities interested in partnering on this prize should submit a response up to 2 pages in length outlining 
their abil ity to deliver on the requirements above. As a part of this response, utilities are invited to request 
changes, suggestions, or clarifications for their partnership on this prize.  
Submit responses by Tuesday, July 26. 
"""

text_2 = """
THIS IS A SOURCES SOUGHT NOTICE ONLY.  This Sources Sought Notice is a means of 
obtaining  feedback  from  contractors  concerning  the requirements  stated  below,  under  Description  of 
the Requirement, which is part of the acquisition planning and market research phase for this 
procurement. The industry input received from this Sources Sought Notice will aid in determining 
available sources that can meet the Government’s below stated requirements and may be considered in 
the preparation of the official sol icitation package.    
 
This Sources Sought Notice does not constitute a solicitation.   Instead, it is an opportunity for the 
Department of Energy (DOE) to maximize small business participation in the Energy Information 
Administration Omnibus Procurement (EOP) V  by measuring the capacity and competency of small 
business to perform in all areas of this acquisition.  Furthermore, we are interested in the capabilities 
and interest of other than small business in this procurement. Previous RFIs have been posted to 
SAM.gov and shared v ia GSA’s Market Research as a Service.  
 
We anticipate the release of the Request for Proposals in the 3rd or 4th quarter of this calendar year.  
 
SUBJECT : Sources  Sought  Notice  for the EOP  V Competition .  
 
INTRODUCTION : 
 
The U.S. Energy Information Administration (EIA) is the statistical and analytical agency within the 
U.S. DOE. EIA collects, analyzes, and disseminates independent and impartial energy information to 
promote sound policymaking, efficient markets, and publi c understanding of energy and its interaction 
with the economy and the environment. EIA is the nation's premier source of energy information and, by 
law, its data, analyses, and forecasts are independent of approval by any other officer or employee of the 
U.S. Government.  
 
To accomplish its mission, EIA conducts a comprehensive data collection program that covers the full 
spectrum of energy sources, end uses, and energy flows.  EIA also prepares informative energy analyses, 
monthly short -term forecasts of energy market trend s, and long -term U.S. and international energy 
outlooks.  EIA disseminates its data products, analyses, reports, and services to customers and 
stakeholders primarily through its website.  EIA has a broad range of stakeholders, with its major 
customers incl uding Congress, Federal and State governments, the private sector, the broader public, 
and the media.  
EIA requires follow -on technical, managerial , and project support  services in the areas of energy 
information and statistical data analysis. Specifically, EIA requires contractor performance in the 
following task areas in order to successfully accomplish these objectives:  
1. Program Management  
2. Survey Operations  
3. Energy Analysis and Modeling  
4. Information Technology Support  
5. Product/Service Development, Production, and Marketing  
6. Organizational Support  
 
If your firm is interested in this requirement and has performed similar projects, we request the 
information  shown  below.  All business  sizes  under  North American Industry Classification System 
(NAICS)  518210  should  consider  submitting  a response to this notice  (note: see Submission Instructions, “D,” below) . 
 
As permitted by Federal Acquisition Regulation (FAR) Part 10, this is a market research tool being 
utilized for informational and planning purposes.  Your responses will assist the Government in the 
development of its acquisition strategy for a possible Request for Proposals (RFP), to be issued at a 
later date,  and in determining  whether  there  are contractors  capable  of completing  these  requirements.  
 
This notice shall not be construed as an RFP or as any commitment or obligation on the part of the 
Government to issue a solicitation.  The Government does not intend to award a contract based on this 
request,  so quotes  will not be considered.  No reimbursement  will be made  for any costs  associated  with 
providing information in response to this synopsis or any follow -up information requests.  
 
Response  is strictly  voluntary  – it is not mandatory  to submit  a response  to this notice  to participate  in 
any formal RFP process that may take place in the future.  However, it should be noted that 
information gathered through this notice may significantly influence our acquisition strategy.  All 
interested parties will  be required  to respond  separately  to any solicitations  posted  as a result of  this 
Sources  Sought Notice.  
 
DESCRIPTION  OF THE  REQUIREMENT:  
 
The anticipated  period  of performance  for EOP V is a 10 -year base with no options.  EOP V will be a 
multiple -award Indefinite Delivery, Indefinite Quantity (IDIQ) contract.  
 
Background:  
EIA’s mission is to collect, analyze, and disseminate independent and impartial energy information to 
promote sound policymaking, efficient markets, and public understanding of energy and its interaction 
with the economy and the environment.  EIA is the Na tion’s premier source of energy information and, 
by law, its data, analyses, and forecasts are independent of approval by any other officer or employee of 
the U.S. Government.  
 
EIA conducts a relevant, reliable, and timely data collection program that covers the full spectrum of 
energy sources, end -uses, and energy flows; generates short - and long -term domestic and international 
energy projections; and performs informative energy  analyses. EIA communicates its statistical and 
analytical products primarily through its website and customer contact center.   
 
Purpose  and Objectives:  
The EIA has the following requirements to fulfill its mission:  
 
1) Develop, manage, operate, and maintain comprehensive energy data collection programs and 
systems.  
2) Develop, manage, operate, and maintain U.S. and international energy models for forecasting 
and performing energy -related analyses.  
3) Disseminate, communicate, and promote the results of these activities to a wide range of 
stakeholders.  
EIA requires contractor performance in the following task areas in order to successfully accomplish 
these objectives:  1. Program Management  
2. Survey Operations  
3. Energy Analysis and Modeling  
4. Information Technology Support  
5. Product/Service Development, Production, and Marketing  
6. Organizational Support  
 
The need for unbiased, statistically -sound energy data information has never been greater and EIA’s 
products and services will continue to help promote sound policy decision making, provide for efficient 
markets, and generate greater public understanding.  Therefore, the primary objective of this acquisition 
is to ensure that EIA acquires the energy -industry and technical resources necessary to accomplish its 
primary mission.  Specifically, EIA plans to achieve its objectives as follows:  
1. Conduct EIA’s data collection programs in the most efficient manner, incorporating best business, 
survey, statistical practices, and standardized methods, introducing innovative solutions, and 
utilizing state -of-the-art technology whenever possible.  
2. Advance EIA’s forecasting and analysis programs with inventive, statistically -sound methods and 
techniques, and knowledge of energy marketplace dynamics.  
3. Continue to develop and deploy electronic products and services using state -of-the-art technology 
and engaging EIA’s wide range of stakeholders on an on -going basis to better understand their 
ongoing and emerging needs.  
4. Improve IT integration support across EIA to streamline services and improve coherence and 
performance.  
5. Conduct contract management activities by continuing to adopt best management practices and 
standardized reporting.  
Project  Requirements  
Description of Service Areas   
 
The following Service Areas  describe the intended focus of this contract:  
 
1. Program Management  
2. Survey Operations  
a. Survey Operations for EIA’s Non -Supply Based Surveys  
b. Survey Operations for EIA’s Supply Based Surveys  
3. Energy Analysis and Modeling  
4. Information Technology Support  
5. Product/Service Development Production and Marketing  
6. Organizational Support  
 
See attachment entitled, “Draft Performance Work Statement  EOP V ,” for the full Project 
Requirements.   
 
SUBMISSION  INSTRUCTIONS:  
 
All interested  contractors  that are certified  small  business  concerns  (please  specify  which  type)  or other  
than small business concerns who believe they can meet the requirements described in this Sources 
Sought Notice are invited to submit, in writing, complete information describing their ability to provide 
the deliverables under this effort.   
Qualified firms shall submit a statement of interest on company letterhead demonstrating the firm’s 
qualifications  to perform  the defined  work.  Responses  must  be complete  and sufficiently  detailed  to 
address the specific information.  The documentation shall address, at a minimum, the following:  
 
A. Company  Profile,  to include:  
1. Company  name  and address;  
2. Affiliate  information,  parent  company,  joint venture  partners,  and potential  
teaming partners;  
3. Year  the firm was established  and number  of employees;  
4. What other procurement vehicle(s) (e.g. GSA MAS, GSA OASIS, CIO -SP3) do you hold 
contracts under and what is the contract number(s)?  
5. Point  of contact  (name,  title, phone  number,  and e-mail address);  
6. UEI number  and CAGE  Code,  as registered  in the System  for Award  Management  
(SAM) at https://www.sam.gov/ , and confirmation of an Active SAM registration.  
7. Business  designation/status  (must  correlate  with SAM  registration)  under  the 
applicable NAICS:  
 
  Small business    HUBZone    WOSB        _____ EDWOSB  
 
  8(a)   VOSB    SDVOSB  
 
  Small Disadvantaged  Business  
 
  Other  than Small Business  
 
B. Documentation  of firm’s  ability/experience:  
1. Statement of which of the 6 EOP V Service  Areas your firm can perform under;  
2. Staff  expertise,  including  their availability,  experience,  and formal  and other  training;  
3. Current  in-house  capability  and capacity  to perform  the work;  
4. Prior  completed  projects  of similar  nature (note which EOP V Service  Area(s) each 
project speaks to);  
5. Corporate  experience  and management  capability;  and 
6. Examples  of prior  completed  Government  contracts,  references,  and other  
related information.  
 
Additionally,  for each project  listed  of same  or similar  nature,  provide  the following:  
1. Brief  description  of the work  performed  and the EOP V Service Area (s) it speaks to ; 
2. Whether your firm was the Prime or a Subcontractor (and if a Subcontractor, what percent 
of the work did your firm perform?);  
3. Project dates (within the last 3 years is preferred);  
4. Project  title; 
5. Dollar  value;  
6. Location;  
7. Customer’s  point  of contact  name  and phone  number.  
 
 
 
 C. Responses  are requested to indicate  the likelihood  of responding  to an actual  solicitation,  if 
an actual solicitation is issued, using the following:  
 
 75% to 100%  
 50% to 74% 
 25% to 49% 
 Less than 25% 
 
D. The anticipated NAICS for this requirement is 518210: Data Processing, Hosting, and Related 
Services (SB size standard: $35 million).  If contractors  believe  a more  appropriate  NAICS  
applies  to the requirement as detailed above , include the feedback in your response.  
 
E. What are your opinions, suggestions, and/or concerns if a full and open (unrestricted) IDIQ 
solicitation, with a 100 % s mall business reserve  for task orders for Service Area  2b, “ Survey 
Operations – Supply Based ;” Service Area  4, “Information Technology Support;” and Service 
Area  5, “Product/Service Development, Production and Marketing ,” was issued?  
 
F. What are your opinions, suggestions, and/or concerns if a partial set -aside IDIQ solicitation, 
with the partial set -aside for small business for Service Area  2b, “ Survey Operations – Supply 
Based ;” Service Area  4, “Information Technology Support;” and Service Area  5, 
“Product/Service Development, Production and Marketing ,” was issued?  
 
G. What are your opinions, suggestions, and/or concerns if the IDIQ solicitation had a 50% reserve 
for small business  for Service Area 1, “Program Management;” Service Area 2a, “Survey 
Operations – Non-supply Based;” Service Area 3, “ Energy Analysis and Modeling ;”  and 
Service Area 6, “ Organizational Support ;” and a 100% reserve for small business  for Service 
Area  2b, “ Survey Operations – Supply Based ;” Service Area  4, “Information Technology 
Support;” and Service Area  5, “Product/Service Development, Production and Marketing ”? 
 
H. For small businesses: What acquisition strategy would increase your firm’s likelihood in 
providing an offer to the RFP?  
 
I. If your firm was interested in this procurement (EOP 1 -4) in the past and your firm did not 
provide an offer, what issues or concerns influence your firm’s decision?  
 
J. Additional  comments or questions:  
 
 
FORMAT OF RESPONSES   
 
Please provide a synopsis of your company’s capabilities and contact information with no more than ten 
(10) single -sided pages, 1 2-point font, minimum one -inch margins, in Microsoft Word format with a 
“.docx” or “.doc” extension or in Adobe format with a “.pdf” extension. To the extent possible, all 
responses should be unclassified and for general access by the Government. Any propri etary information 
submitted must be clearly and separately identified and marked and will be appropriately protected. The 
Gove rnment will NOT be responsible for any proprietary information not clearly marked. Any material 
provided may be used in development of future solicitations . 
 
All responses  shall  be submitted  via e-mail on or before  5:00 PM Eastern  Time  (ET),  May  27, 2022 , 
to the Contracting Officer, Katherine Bowen (e -mail: Katherine.Bowen@hq.doe.gov), and must include 
the information requested above . Late responses will not be accepted.  This is strictly market research and the Government will not entertain any questions.  
 
Respondents  will not be notified  of the results  of the evaluation.  We appreciate  your interest  and thank 
you in advance for responding to th is Sources Sought Notice. 
"""
text_3="""
Request for Information/Sources Sought Notice
Renewable Energy Certificates Registry
Southwestern Power Administration


1. This is a sources sought notice. This is not a request for proposal, a request for quote nor a request for full qualifications packages. This information is for market research and planning purposes only.

The purpose of this sources sought is to conduct market research to identify if responsible sources exist, and to assist in determining if this effort should be set-aside for small business concerns. The proposed North American Industry Classification Systems (NAICS) Code is 221122. The Government will use this information to determine the best acquisition strategy for this procurement. The Government is interested in all small businesses to include 8(a), Service-Disabled Veteran-Owned, HUBZone, and Women-Owned Small Business concerns with the capability to perform this service.
2. Responses to this Sources Sought/RFI shall be limited to five (5) pages in a single PDF document and should include the following information (item numbers (1) through (4) below). Additional pages, cover letters, and extraneous materials are NOT desired. Materials, documents, web links, etc. incorporated by reference in responses will not be accessed by the Government for inclusion; all responses must be self-contained, to address the following items.

(1) Firm name, address, point of contact, phone number, and e-mail address.

(2) Firm Unique Entity Identifier (UEI) and CAGE Code. In order to receive a federal contract, you must have an active registration in SAM.

(3) Firm business size in relation to NAICS code 221122, which has a small business size standard of 1100 Employees in annual average gross receipts over the past five (5) years. Identify whether your firm is a Large/Small business. If small, identify small business type(s) (HUBZone, SDVOSB, 8(a), WOSB, etc.)

(4) Experience. Demonstration of the firm’s experience as a prime contractor executing the work below in section 3. Identify if your product meets the requirements outlined in “REC Registry Requirements”.
3. Requirement
Background:
As one of four Power Marketing Administrations in the United States, Southwestern Power Administration (SWPA) markets renewable hydroelectric power from Corps-owned multipurpose water resource projects located in Arkansas, Missouri and Oklahoma. The power is marketed to 78 municipal utilities, 21 rural electric cooperatives, and 3 military installations in those four states as well as in Kansas and Louisiana. Those customers distribute the power to approximately 10 million end users in the six-state area. The estimated average annual energy of all 19 projects is 4,757 gigawatt-hours.
For SWPA to implement a renewable energy certificate (REC) program for its customers, SWPA will require an account with a REC tracking system or registry. SWPA’s key objective in implementing a REC program is subscribing to a REC tracking system or registry to enable the creation of RECs and facilitate the management and retirement of RECs.
Overview:
Concerns about climate change have driven a nationwide push toward clean, renewable energy production such as hydropower. Many states have adopted renewable portfolio standards which require companies to provide a certain percentage of their electricity from renewable sources by a specified year. Kansas, Missouri, and Texas in SWPA’s marketing area have adopted renewable portfolio standards. Other states have adopted non-binding renewable energy goals.
RECs are tradable, non-physical commodities in the energy market that represent the benefits associated with 1 megawatt-hour (MWh) of generated renewable energy. RECs enable the REC owner to claim the environmental benefits – the reduced carbon footprint – of that clean energy.
There are two main markets for renewable energy certificates in the United States – compliance markets and voluntary markets. Compliance markets are created in states with renewable portfolio standards. Electric utilities in these states demonstrate compliance with their requirements by purchasing RECs. Voluntary markets are ones in which customers choose to buy renewable power out of a desire to use renewable energy. Renewable energy generators located in states that do not have a renewable portfolio standard can sell their RECs to voluntary buyers.
A tracking system (or registry) is a database, in this case a web or cloud based electronic service, with basic information about each MWh generated from generation facilities registered in the database. Electronic tracking systems allow RECs to be transferred among account holders much as in online banking. The database tracks certain information for each MWh, including facility location, generation technology, facility owner, fuel type, capacity, the year the facility began operating, and the month/year the MWh was generated. Each REC is issued a unique serial ID number and can only be held in one account at a time.
Scope of Work: To implement a REC program, SWPA must subscribe with a REC tracking system or registry to enable the creation of RECs and facilitate the management and retirement of RECs.
REC Registry Requirements:
A REC registry suitable for use by SWPA must meet the following requirements:
1. Privacy: User information will be confidential and accessible only through secured user login credentials.
2. Multiple Users: account allows SWPA to create additional users within the organization that each have their own unique user ID and password. Also, allows for setting certain privileges based on the user’s need.
3. IT Compliance. Must follow SWPA Information Technology requirements for web-based service platforms and data storage. The REC registry web site must be accessible through Google Chrome or Microsoft Edge. See https://www.energy.gov/management/pf-2021-36-chief-information-officers-supply-chain-risk-management-program for more information.
4. REC Account. The REC registry will provide SWPA an account and login credentials for its web-based or cloud-based service. That account will facilitate the registration of SWPA’s 19 projects in the region and the management of all RECs related to generation from SWPA’s 19 projects.
5. REC Creation/Issuance. Must provide for creation of RECs from the renewable energy generated at SWPA’s 19projects located in the states of Arkansas, Missouri and Oklahoma. When the REC registry account is created, SWPA will provide the most recent two years of generation data from its projects for REC creation. Subsequently, SWPA will provide the meter data from the Corps projects on a monthly basis for the creation of the RECs through a tabular format that can be uploaded to the registry through the web accessible application. Alternatively, the registry will have the capability to receive the generation data directly from Midcontinent Independent System Operator (MISO) or Southwest Power Pool (SPP) on a monthly basis to facilitate the REC creation. SWPA understands that a release form from SWPA will likely be required for the upload of data from MISO or SPP to the registry. The number of RECs will represent the number of MWhs of net generation from SWPA’s 19 projects. Once a REC is created, it will remain in SWPA’s registry account until transferred, retired, or exported on behalf of a SWPA customer. Issue one electronic Certificate for each MWh of energy generated by those Generating Units registered. A Certificate created and tracked within the registry will represent all renewable attributes from one MWh of renewable generation. At a minimum fields that will be included on each certificate are:
a. Serial Number(s)
b. Account
c. ID
d. Generator Fuel Type
e. Vintage Date
f. Location
g. Quantity
h. Eligibilities

6. REC Transfer/Transaction. Must provide the ability for SWPA to transfer RECs to SWPA customer accounts, including export to other REC registries. Once the RECs are created within SWPA’s registry account, the registry will allow SWPA to transfer RECs to a customer’s account on a periodic monthly basis. SWPA will determine the number of RECs available for transfer to each customer and will communicate that information to customers participating in the REC program. SWPA will facilitate the transfer of the RECs from SWPA’s registry account to the customer’s registry account. Must have option to transfer active certificates to 1) Another organization, 2) Another active account, 3) to a compatible tracking system. From initiation of transaction by SWPA to received and active certificates by the customer should take no longer than 7 business days on average. Transactions will be configured to prevent duplication.
7. REC Retirement. Must provide ability for SWPA to retire RECs on behalf of SWPA customers.
Once the RECs are created within SWPA’s registry account, the registry will allow SWPA to retire RECs on a customer’s behalf on a monthly basis. SWPA will determine the number of RECs available for retirement for each customer and will communicate that information to customers. Upon the communicated desire of a customer to retire their designated RECs, SWPA will facilitate the retirement of those RECs. Once retired, a REC can no longer be transferred or exported. Retirements certificates will provide additional attributes based on its type. Types of voluntary retirement will include:
a. Municipal Renewable Portfolio Standard (Compliance) (MUN)
b. Federal Renewable Energy Requirement (FDR)
c. State-Regulated Utility Renewable Portfolio Standard / Provincial Utility Portfolio Standard (Compliance) (RPS)
8. REC Export to Other Registries. Must be able to export RECs to other registries. Some SWPA customers may participate in a different registry than SWPA and may wish their designated RECs to be exported to that registry. Once the RECs are created within SWPA’s account with the registry, the registry will allow SWPA to export RECs to a customer’s account in a different registry on a monthly basis. SWPA will determine the number of RECs available for export to those customers and will communicate that information to the customers. Upon the concurrence of each customer, SWPA will facilitate the export of the RECs to the customer’s desired registry.
9. Reporting Capabilities. Must provide reporting capabilities so that SWPA can easily produce information on REC activities by project or customer on a monthly or other user-specified timeframe. Such reports are necessary to provide to the customers at periodic (SPRA) meetings or on request by customers.
10. Registry Accessibility. Must be recognized by states which have renewable portfolio standards and be able to function in both compliance and voluntary markets.
11. State Compliance: The registry must be recognized by states which have renewable portfolio standards including MO, KS, LA.
12. Account Administrator. SWPA will designate an Account Administrator for the account with the REC registry who will be SWPA’s main point of contact with the registry. The Account Administrator will work with the REC registry in the creation of the REC registry account and the subsequent creation and management of RECs associated with SWPA project generation. The Account Administrator will also have the ability to designate Account Users to access and utilize the REC registry account.
13. Scope. Based on past power production and future estimates, SWPA anticipates the creation and retirement of between 12-15 million RECs in its first full year of implementation and between 4-5 million RECs per year thereafter. The first year’s expanded scope is due to the reclamation of RECs from the prior two years.

Points of Contact:
Brooke Butcher; brooke.butcher@swpa.gov
Erica Wilson; erica.wilson@swpa.gov
"""
summary_with_instruction = instruction + summary
embedding_summary = get_embedding(summary_with_instruction)
embeddings_text = [get_embedding(text_1), get_embedding(text_2), get_embedding(text_2)]

for i, embedding_selected_text in enumerate(embeddings_text):
    print(f"{i+1}번째 선택된 텍스트와의 유사도: {round(cosine_similarity(embedding_summary, embedding_selected_text)*100, 2)}") # Extract the matching scores for each RFP description. 
    print("-"*100)