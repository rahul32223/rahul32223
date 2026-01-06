import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------- Sample Data --------------------
candidates = pd.DataFrame([
    {"id": "C001", "email": "c1@example.com", "skills": "python, machine learning", "location": "Delhi", "category": "SC", "past_participation": 0},
    {"id": "C002", "email": "c2@example.com", "skills": "data analysis, sql", "location": "Mumbai", "category": "OBC", "past_participation": 1},
    {"id": "C003", "email": "c3@example.com", "skills": "web development, javascript", "location": "Delhi", "category": "ST", "past_participation": 0}
])

internships = pd.DataFrame([
    {"id": "I101", "required_skills": "python, deep learning", "location": "Delhi", "capacity": 1},
    {"id": "I102", "required_skills": "sql, dashboarding", "location": "Mumbai", "capacity": 2},
    {"id": "I103", "required_skills": "javascript, react", "location": "Delhi", "capacity": 1}
])

use_affirmative_action = True

# -------------------- Email Function --------------------
def send_email(to_email, candidate_id, internship_id):
    sender_email = "your_email@example.com"
    sender_password = "your_app_password"  # Use app password if using Gmail

    subject = "Internship Match Notification"
    body = f"""
    Dear Candidate {candidate_id},

    Congratulations! You have been matched to Internship {internship_id}.
    Please log in to the portal for more details.

    Regards,
    Internship Matching Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email failed to send to {to_email}: {e}")
        return False

# -------------------- Matching Logic --------------------
def match_candidates(candidates, internships, similarity_scores):
    matches = []
    for i, candidate in candidates.iterrows():
        best_score = -1
        best_match = None
        for j, internship in internships.iterrows():
            score = similarity_scores[i][j]
            if use_affirmative_action:
                if candidate['location'].lower() == internship['location'].lower():
                    score += 0.2
                if candidate['category'] in ['SC', 'ST']:
                    score += 0.1
                if candidate['past_participation'] == 0:
                    score += 0.1
            if internship['capacity'] > 0 and score > best_score:
                best_score = score
                best_match = internship['id']
        if best_match:
            matches.append({
                'Candidate ID': candidate['id'],
                'Internship ID': best_match,
                'Match Score': round(best_score, 3)
            })
            internships.loc[internships['id'] == best_match, 'capacity'] -= 1
            send_email(candidate['email'], candidate['id'], best_match)
    return pd.DataFrame(matches)

# -------------------- Skill Vectorization --------------------
vectorizer = CountVectorizer()
skill_matrix = vectorizer.fit_transform(
    candidates['skills'].astype(str).tolist() + internships['required_skills'].astype(str).tolist()
)
candidate_vectors = skill_matrix[:len(candidates)]
internship_vectors = skill_matrix[len(candidates):]
similarity_scores = cosine_similarity(candidate_vectors, internship_vectors)

# -------------------- Run Matching --------------------
matched_df = match_candidates(candidates, internships, similarity_scores)
print("\n🔍 Matching Results:")
print(matched_df)

# -------------------- Manual Match Override --------------------
manual_match = {
    'Candidate ID': "C002",
    'Internship ID': "I103",
    'Match Score': 'Manual Override'
}
matched_df = pd.concat([matched_df, pd.DataFrame([manual_match])], ignore_index=True)
print("\n🛠️ Manual Match Override Applied:")
print(matched_df)

# -------------------- Save to CSV --------------------
matched_df.to_csv("internship_matches.csv", index=False)
print("\n📥 Matches saved to internship_matches.csv")