import sqlite3
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PaperDBChat:
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_database_stats(self) -> Dict[str, Any]:
        """Get various statistics about the database."""
        cursor = self.conn.cursor()

        # Get total number of papers
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        total_papers = cursor.fetchone()['count']

        # Get number of papers by year
        cursor.execute("""
            SELECT substr(date, 1, 4) as year, COUNT(*) as count 
            FROM papers 
            GROUP BY year 
            ORDER BY year DESC
        """)
        papers_by_year = {row['year']: row['count']
                          for row in cursor.fetchall()}

        # Get average scores
        cursor.execute("""
            SELECT 
                AVG(relevance) as avg_relevance,
                AVG(clarity) as avg_clarity,
                AVG(novelty) as avg_novelty,
                AVG(significance) as avg_significance
            FROM papers
        """)
        avg_scores = dict(cursor.fetchone())

        # Get most common tags
        cursor.execute("""
            SELECT tags, COUNT(*) as count
            FROM papers
            WHERE tags IS NOT NULL AND tags != ''
            GROUP BY tags
            ORDER BY count DESC
            LIMIT 5
        """)
        top_tags = {row['tags']: row['count'] for row in cursor.fetchall()}

        return {
            "total_papers": total_papers,
            "papers_by_year": papers_by_year,
            "avg_scores": avg_scores,
            "top_tags": top_tags
        }

    def handle_meta_question(self, question: str) -> Tuple[bool, str]:
        """Handle questions about the database itself."""
        try:
            # Get database statistics
            stats = self.get_database_stats()

            # Create context for GPT
            context = f"""You are a helpful assistant that answers questions about a database of academic papers.
            Here are the current statistics:
            - Total number of papers: {stats['total_papers']}
            - Papers by year: {stats['papers_by_year']}
            - Average scores:
              * Relevance: {stats['avg_scores']['avg_relevance']:.2f}
              * Clarity: {stats['avg_scores']['avg_clarity']:.2f}
              * Novelty: {stats['avg_scores']['avg_novelty']:.2f}
              * Significance: {stats['avg_scores']['avg_significance']:.2f}
            - Top tags: {stats['top_tags']}

            Please answer the user's question about the database using these statistics.
            If the question is not about the database itself, respond with 'NOT_A_META_QUESTION'."""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )

            answer = response.choices[0].message.content.strip()
            return answer != "NOT_A_META_QUESTION", answer

        except Exception as e:
            return False, f"Error handling meta question: {str(e)}"

    def generate_search_terms(self, natural_query: str) -> List[str]:
        """Convert natural language query into search terms using GPT."""
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that converts natural language queries about academic papers into relevant search terms.
                    Extract key concepts, topics, and technical terms that would be useful for searching academic papers.
                    Return a list of 3-5 most relevant search terms, separated by commas."""},
                    {"role": "user", "content": natural_query}
                ],
                temperature=0.1
            )
            terms = response.choices[0].message.content.strip().split(',')
            return [term.strip() for term in terms]
        except Exception as e:
            print(f"Warning: Error generating search terms: {str(e)}")
            return [natural_query]  # Fallback to original query

    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search papers in the database using natural language processing."""
        cursor = self.conn.cursor()

        # Generate search terms from natural language query
        search_terms = self.generate_search_terms(query)
        print(f"\n🔍 Searching for: {', '.join(search_terms)}")

        # Build the SQL query with multiple search terms
        conditions = []
        params = []
        for term in search_terms:
            conditions.append("""
                (title LIKE ? OR authors LIKE ? OR summary LIKE ? OR tags LIKE ?)
            """)
            params.extend([f"%{term}%"] * 4)

        sql_query = f"""
            SELECT id, title, authors, summary, tags, date, relevance, clarity, novelty, significance
            FROM papers
            WHERE {' OR '.join(conditions)}
            ORDER BY relevance DESC, date DESC
            LIMIT 5
        """

        cursor.execute(sql_query, params)

        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        return results

    def get_paper_details(self, paper_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT *
            FROM papers
            WHERE id = ?
        """, (paper_id,))

        row = cursor.fetchone()
        return dict(row) if row else None

    def create_chat_context(self, paper: Dict[str, Any]) -> str:
        """Create a context string for the chat model."""
        context = f"""Paper Title: {paper['title']}
Authors: {paper['authors']}
Date: {paper['date']}
Tags: {paper['tags']}
Relevance Score: {paper['relevance']}
Clarity Score: {paper['clarity']}
Novelty Score: {paper['novelty']}
Significance Score: {paper['significance']}

Summary:
{paper['summary']}

You are a helpful AI assistant that answers questions about this academic paper. 
Please provide clear, accurate, and concise answers based on the paper's content.
If the answer cannot be found in the paper, say so clearly.
When answering:
1. Be specific and cite relevant parts of the paper
2. Explain technical terms if they might be unfamiliar
3. If the answer is not in the paper, say so clearly
4. If you're unsure about something, acknowledge the uncertainty"""
        return context

    def chat_with_paper(self, paper_id: int, question: str) -> str:
        """Chat with the paper using GPT."""
        paper = self.get_paper_details(paper_id)
        if not paper:
            return "❌ Paper not found."

        context = self.create_chat_context(paper)

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error getting response from GPT: {str(e)}"


def main():
    print("🤖 Welcome to the Paper Database Chat Bot!")
    print("\nYou can ask questions like:")
    print("- 'How many papers do you have in the database?'")
    print("- 'What are the most common topics in your papers?'")
    print("- 'Show me papers about machine learning in healthcare'")
    print("- 'What papers discuss climate change solutions?'")
    chat = PaperDBChat()

    while True:
        # Get query
        query = input(
            "\n🔍 What would you like to know? (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break

        if not query:
            continue

        # First, check if it's a meta question about the database
        is_meta, meta_answer = chat.handle_meta_question(query)
        if is_meta:
            print(f"\n📊 {meta_answer}")
            continue

        # If not a meta question, treat it as a paper search
        print("\n📚 Searching papers...")
        results = chat.search_papers(query)

        if not results:
            print("❌ No papers found matching your query.")
            continue

        # Display results
        print("\n📖 Found papers:")
        for i, paper in enumerate(results, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Date: {paper['date']}")
            print(f"   Relevance: {paper['relevance']}")

        # Select paper
        while True:
            try:
                choice = input(
                    "\n📄 Enter paper number to chat (or 'back' to search again): ").strip()
                if choice.lower() == 'back':
                    break
                if choice.lower() == 'quit':
                    return

                paper_idx = int(choice) - 1
                if 0 <= paper_idx < len(results):
                    selected_paper = results[paper_idx]
                    print(f"\n📚 Selected: {selected_paper['title']}")

                    # Chat loop for this paper
                    while True:
                        question = input(
                            "\n❓ Your question (or 'back' for new search): ").strip()
                        if question.lower() == 'back':
                            break
                        if question.lower() == 'quit':
                            return

                        if not question:
                            continue

                        print("\n🤔 Thinking...")
                        answer = chat.chat_with_paper(
                            selected_paper['id'], question)
                        print(f"\n📝 Answer: {answer}")
                else:
                    print("❌ Invalid paper number. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")


if __name__ == "__main__":
    main()
