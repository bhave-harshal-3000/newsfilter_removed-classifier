import os
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from emailer import send_email, build_html_email
from higher_ed import scrape_higher_ed_news
from entertainment import scrape_entertainment_news
from sports import scrape_sports_news
from business_and_finance import scrape_business_finance_news
from dotenv import load_dotenv
from technology import scrape_technology_news
from news_sources import scrape_news
from environment import scrape_environment_news
from industry import scrape_industry_news
from health import scrape_health_news


load_dotenv(dotenv_path="scratch.env")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


# Add this mapping at the top of your file
SPORTS_SOURCE_MAP = {
    "indian_express_sports": "Indian Express",
    "espncricinfo": "ESPN Cricinfo",
    "ndtv_sports": "NDTV Sports",
    "the_hindu_sports": "The Hindu",
    "bbc_sport": "BBC Sport",
    "guardian_sports": "The Guardian",
    "espn_global": "ESPN"
}


def select_top_news_with_gemini(articles, top_n=10, return_scores=False):
    print(f"[Gemini] Preparing to call Gemini LLM with {len(articles)} articles, requesting top {top_n}.")
    if not GEMINI_API_KEY:
        print("Gemini API key not found.")
        if return_scores:
            return [(art, None) for art in articles[:top_n]]
        return articles[:top_n]

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash", google_api_key=GEMINI_API_KEY
    )
    prompt = (
        "You are an expert news assistant. "
        "Given the following list of news headlines (with their sources and links), "
        "for each headline, assign an importance score from 1 (least important) to 10 (most important) for inclusion in an education news digest. "
        "Consider newsworthiness, impact, and diversity. "
        "Return your answer as a numbered list in this format:\n"
        "<SOURCE>, <HEADLINE>\n<LINK>\nScore: <score>\n\n"
        "Here is the list:\n"
    )
    for idx, article in enumerate(articles, 1):
        source = article.get("source", "Unknown Source")
        title = article["title"]
        url = article["url"]
        prompt += f"{idx}. {source}, {title}\n{url}\n"

    print("[Gemini] Calling Gemini LLM API...")
    response = llm.invoke([HumanMessage(content=prompt)])
    print("[Gemini] Gemini LLM API call completed.")
    print("Gemini raw output:\n", response.content)  # For debugging

    lines = str(response.content).split("\n")
    import re

    scored_articles = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and "," in line and line[0].isdigit() and "." in line:
            source_headline = line.split(".", 1)[1].strip()
            url = lines[i + 1].strip() if i + 1 < len(lines) else ""
            score_line = lines[i + 2].strip() if i + 2 < len(lines) else ""
            score_match = re.search(r"Score:\s*(\d+)", score_line)
            score = int(score_match.group(1)) if score_match else 0
            for art in articles:
                if art["title"] in source_headline and art["url"] in url:
                    scored_articles.append((art, score))
                    break
            i += 3
        else:
            i += 1

    # Sort by score descending
    scored_articles.sort(key=lambda x: x[1], reverse=True)

    # Print all scores in descending order
    # print("[Gemini] All scored articles (descending order):")
    # for idx, (art, score) in enumerate(scored_articles, 1):
    #     print(f"[{idx}] Score: {score} | Title: {art.get('title')} | Source: {art.get('source')} | URL: {art.get('url')}")

    # Return top N
    if return_scores:
        return scored_articles[:top_n]
    else:
        return [art for art, score in scored_articles[:top_n]]


from transformers import pipeline

import smtplib
from email.mime.text import MIMEText
import re



# --- Enhanced Classifier ---
try:
    classifier = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    print("Classifier loaded successfully")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None


def clean_title(title):
    if not title:
        return None
    title = re.sub(r"\s+", " ", title.strip())
    if len(title) < 10 or len(title) > 200:
        return None
    # Filter out navigation items and ads
    skip_words = [
        "subscribe",
        "login",
        "register",
        "advertisement",
        "menu",
        "search",
        "newsletter",
    ]
    if any(word in title.lower() for word in skip_words):
        return None
    return title


def classify_article(title):
    try:
        if not classifier:
            # Fallback classification based on keywords
            sensitive_keywords = [
                "scandal",
                "controversy",
                "crisis",
                "protest",
                "violence",
                "death",
                "suicide",
                "harassment",
                "discrimination",
                "abuse",
                "fraud",
                "corruption",
                "strike",
                "riot",
                "attack",
                "murder",
                "assault",
                "drugs",
                "alcohol",
                "ragging",
                "sexual",
                "assault",
                "mental health crisis",
                "depression",
                "anxiety",
                "rape",
            ]

            title_lower = title.lower()
            for keyword in sensitive_keywords:
                if keyword in title_lower:
                    return "Sensitive"
            return "General"

        # Use sentiment analysis
        result = classifier(title)[0]

        sensitive_keywords = [
            "scandal",
            "controversy",
            "crisis",
            "protest",
            "violence",
            "death",
            "suicide",
            "harassment",
            "discrimination",
            "abuse",
            "fraud",
            "corruption",
            "strike",
            "riot",
            "attack",
            "murder",
            "assault",
            "drugs",
            "alcohol",
            "ragging",
            "rape",
        ]

        title_lower = title.lower()
        for keyword in sensitive_keywords:
            if keyword in title_lower:
                return "Sensitive"

        if result["label"] == "LABEL_0" and result["score"] > 0.8:  # Very negative
            return "Sensitive"

        return "General"

    except Exception as e:
        print(f"Error classifying article '{title}': {e}")
        return "General"


def filter_articles(articles, content_type):
    result = []
    print(f"Filtering {len(articles)} articles for content type: {content_type}")

    for article in articles:
        category = classify_article(article["title"])
        print(f"Article: '{article['title'][:50]}...' -> {category}")

        if content_type == "All":
            result.append((article, category))
        elif content_type == "Only General" and category == "General":
            result.append((article, category))
        elif content_type == "Only Sensitive" and category == "Sensitive":
            result.append((article, category))

    print(f"Filtered to {len(result)} articles")
    return result


def create_display_url(url, max_length=50):
    if len(url) <= max_length:
        return url

    if "news.google.com" in url:
        return f"news.google.com/articles/... (Google News)"

    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        if len(domain) > max_length - 10:
            return f"{domain[:max_length - 7]}..."

        # Show domain + truncated path
        path = parsed.path
        remaining_space = max_length - len(domain) - 3

        if len(path) > remaining_space:
            return f"{domain}{path[:remaining_space]}..."
        else:
            return f"{domain}{path}"

    except:
        # Fallback: just show beginning and end
        if len(url) > max_length:
            half = (max_length - 5) // 2
            return f"{url[:half]} ... {url[-half:]}"
        return url


def format_email(articles):
    if not articles:
        return "No education news articles found for your preferences today.\n\nPlease try again later or adjust your preferences."

    general = []
    sensitive = []

    for article, category in articles:
        if category == "General":
            general.append(article)
        else:
            sensitive.append(article)

    body = "🎓 Your Education News Digest 🎓\n"
    body += "=" * 50 + "\n\n"

    if general:
        body += "📚 GENERAL NEWS:\n"
        body += "-" * 20 + "\n"
        for i, article in enumerate(general, 1):
            display_url = create_display_url(article["url"])
            body += f"{i}. {article['title']}\n"
            body += f"   🔗 Full link: {article['url']}\n\n"

    if sensitive:
        body += "⚠️  SENSITIVE NEWS:\n"
        body += "-" * 20 + "\n"
        for i, article in enumerate(sensitive, 1):
            display_url = create_display_url(article["url"])
            body += f"{i}. {article['title']}\n"
            body += f"   🔗 Full link: {article['url']}\n\n"

    if not general and not sensitive:
        body += "No articles match your content preferences today.\n"

    body += "\n" + "=" * 50 + "\n"
    body += "This digest was automatically generated.\n"
    body += "Stay informed, stay educated! 🌟"

    return body


def process_and_send(emails, category, region, content_type, top_n=10, sources=None):
    
    print(f"[process_and_send] Function Called with \
            category={category}, \
            region={region}, \
            content_type={content_type}, \
            top_n={top_n}, \
            sources={sources}")

    errors = []
    articles = []
    topic = ""

    # Scraping step
    print(f"Starting scraping for category: {category}")
    print(f"Sources to scrape: {sources}")
    if category == "higher_ed":
        articles = scrape_higher_ed_news(region=region, sources=sources)
        topic = f"{region} Higher Education"
    elif category == "entertainment":
        articles = scrape_entertainment_news(region=region, sources=sources)
        topic = f"{region} Entertainment"
    elif category == "sports":
        articles = scrape_sports_news(region=region, sources=sources)
        topic = f"{region} Sports"
    elif category == "business_and_finance":
        articles = scrape_business_finance_news(region=region, sources=sources)
        topic = f"{region} Business & Finance"
    elif category == "tech":
        articles = scrape_technology_news(region=region, sources=sources)
        topic = f"{region} Technology"
    elif category == "environment":
        articles = scrape_environment_news(region=region, sources=sources)
        topic = f"{region} Environment"
    elif category == "industry":
        articles = scrape_industry_news(region=region, sources=sources)
        topic = f"{region} Industry"
    elif category == "health":
        articles = scrape_health_news(region=region, sources=sources)
        topic = f"{region} Health"

    else: # Default to general
        articles, errors = scrape_news(region, sources)
        topic = region if region else "General"
    print(f"[process_and_send] Scraping complete. Found {len(articles)} articles.")

    # Support multiple emails separated by comma or semicolon
    if not emails:
        msg = "\u274c Please enter at least one email address"
        print("[process_and_send] No emails provided.")
        return msg

    # Split and clean emails
    email_list = [e.strip() for e in re.split(r"[;,]", emails) if e.strip()]
    invalids = [e for e in email_list if "@" not in e]
    if not email_list or invalids:
        msg = f"\u274c Invalid email(s): {', '.join(invalids)}"
        print(f"[process_and_send] Invalid emails: {invalids}")
        return msg

    if not articles:
        print("[process_and_send] No articles found after scraping.")
        msg = "\u26a0\ufe0f No articles found for the selected region. Please try again later."
        if errors:
            msg += "\n\n\u26a0\ufe0f Some sources failed to scrape:\n" + "\n".join(errors)
        return msg

    # Filter articles by content type (General/Sensitive/All)
    
    filtered = []
    for article in articles:
        category_label = classify_article(article["title"])
        if content_type == "All":
            filtered.append(article)
        elif content_type == "Only General" and category_label == "General":
            filtered.append(article)
        elif content_type == "Only Sensitive" and category_label == "Sensitive":
            filtered.append(article)

    

    # Use Gemini to select top N important headlines
    print(f"Calling select_top_news_with_gemini with {len(filtered)} articles.")
    top_articles = select_top_news_with_gemini(filtered, top_n=top_n)
    print(f"Gemini selection complete. {len(top_articles)} articles selected.")

    # Format email body once
    email_body = format_email([(a, classify_article(a["title"])) for a in top_articles])
    html_body = build_html_email(top_articles, topic=topic)
    subject = f" {topic} News Digest - (Top {top_n} articles)"

    # Send to all emails
    success, failed = [], []
    for email in email_list:
        if send_email(email, subject, email_body, html_body):
            success.append(email)
        else:
            failed.append(email)

    msg = ""
    if success:
        msg += f"\u2705 Email sent successfully to: {', '.join(success)}\n"
    if failed:
        msg += f"\u274c Failed to send email to: {', '.join(failed)}"
    if errors:
        msg += "\n\n\u26a0\ufe0f Some sources failed to scrape:\n" + "\n".join(errors)
    print(f"Done. Returning status message.")
    return msg.strip()

# import gradio as gr


# # Create Gradio UI
# with gr.Blocks(title="Education News Digest") as demo:
#     gr.Markdown("# 📰 Education News Digest")
#     gr.Markdown("Get personalized education news articles delivered directly to your email.")

#     with gr.Row():
#         with gr.Column():
#             email = gr.Textbox(label="Your Email", placeholder="example@email.com")
#             region = gr.Radio(["India", "Global"], label="News Region", value="India")
#             content_type = gr.Radio(
#                 ["All", "Only General", "Only Sensitive"],
#                 label="Content Type",
#                 value="All",
#                 info="Choose whether to receive all articles or filter by content type"
#             )
#             top_n = gr.Radio(
#                 [10, 20, 30],
#                 label="I want:",
#                 value=10,
#                 info="Number of top news articles to receive"
#             )
#             submit_btn = gr.Button("Send Me News!", variant="primary")

#         with gr.Column():
#             output = gr.Textbox(label="Status", interactive=False)


#     submit_btn.click(
#         fn=process_and_send,
#         inputs=[email, region, content_type, top_n],
#         outputs=output,
#     )
# if __name__ == "__main__":
#     demo.launch(share=True)
