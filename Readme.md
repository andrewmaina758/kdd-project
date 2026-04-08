# KDD Project: Student Job Recommendation

This repository contains a notebook for matching students to job opportunities. The project cleans a large job-posting dataset, enriches it with ZIP-code-based location data, converts both jobs and student profiles into TF-IDF vectors, and uses cosine similarity to recommend career directions that look most relevant for a given student.

The core logic currently lives inside [data_cleaning.ipynb](data_cleaning.ipynb). In other words, this project is not a packaged application yet; it is an exploratory pipeline that combines data cleaning, feature engineering, retrieval, and simple recommendation reporting in one notebook.

## What The Project Is Doing

At a high level, the notebook is trying to answer three questions:

1. Which jobs look most similar to a student's background?
2. Which job titles appear repeatedly among those strong matches?
3. In which states are those matched roles most in demand?

Instead of hand-writing rules such as "computer science students should only see software roles," the notebook learns a text representation from real job postings. It then projects student profiles into that same text space and asks: "Which jobs use similar language to this student's major, skills, experience area, degree, and school year?"

## Repository Structure

- [data_cleaning.ipynb](data_cleaning.ipynb): the main workflow for data cleaning and recommendation logic
- [data/uszips.csv](data/uszips.csv): ZIP-to-city/state lookup table used to enrich job locations
- `data/postings.csv`: required local job-posting dataset expected by the notebook, but not currently committed to this repository
- [requirements.txt](requirements.txt): Python dependencies used by the notebook

## Deep Walkthrough Of The Notebook

### 1. Load and narrow the raw job-posting dataset

The notebook starts by reading `data/postings.csv` and keeping only the columns needed for the recommendation pipeline:

- `job_id`
- `company_name`
- `title`
- `description`
- `location`
- `company_id`
- `views`
- `skills_desc`
- `work_type`
- `zip_code`
- `fips`

Important detail: `job_id`, `company_id`, `zip_code`, and `fips` are read as strings. This matters because ZIP codes and FIPS codes can contain leading zeros, and converting them to integers would destroy that information.

It also lowercases and trims the main text columns so later matching steps are less sensitive to formatting differences.

From the notebook output, the raw jobs table begins with:

- `123,849` rows
- `11` columns

### 2. Remove unusable generic locations

Some jobs have `location = "united states"` instead of a usable city or state. The notebook treats those rows carefully:

- If a row says only `united states`
- And it has no ZIP code
- And it has no FIPS code

then that row is dropped because the project cannot reliably place it geographically.

The notebook reports that this step drops `8,125` rows.

### 3. Enrich jobs with city and state from ZIP codes

The file [data/uszips.csv](data/uszips.csv) is used as a lookup table. The notebook:

- standardizes column names
- finds the appropriate state column (`state`, `state_id`, or `state_name`)
- zero-pads ZIP codes to five digits
- merges job postings with the ZIP dataset

After that merge, many jobs gain `city` and `state` fields even if the original `location` text was messy.

At this stage, the notebook output shows:

- `115,724` job rows remaining
- `14` columns, including added `zip`, `city`, and `state`

### 4. Recover missing states from the free-text location field

Not every row can be resolved from ZIP alone. To recover additional missing states, the notebook defines a `STATE_MAP` and scans the free-text `location` column for:

- full state names such as `texas`
- abbreviations such as `TX`

This is a practical fallback step. If ZIP-based enrichment fails but the location text still contains `Dallas, TX` or `Austin, Texas`, the project can still recover the state.

### 5. Drop rows that still cannot be placed

After ZIP enrichment and location-string recovery, the notebook drops any rows whose state is still missing. This is an important design choice: the project prefers a smaller, geographically reliable dataset over a larger but ambiguous one.

The final cleaned job table shown in the notebook has:

- `111,748` rows
- `13` columns

This cleaned dataset is the working base for the recommender.

### 6. Load and normalize the student dataset

The student dataset is loaded from a remote CSV hosted on GitHub:

`https://raw.githubusercontent.com/andrewmaina758/job-recommendation-system/refs/heads/main/students_dataset.csv`

The notebook then lowercases the string columns to keep matching consistent.

The notebook output shows the student data has:

- `600` student rows
- `10` columns

The fields used include:

- `Academic Major`
- `GPA`
- `Skill`
- `Location Interest`
- `University`
- `School Year`
- `Area of Experience`
- `Number of Experience (yrs)`
- `Degree`

### 7. Build job text for semantic matching

The notebook creates a `job_text` field by combining:

- job title
- job description

This gives the model one text block per job posting that roughly captures what the role is and what it asks for.

### 8. Normalize job titles for role-level grouping

The recommender is not satisfied with only showing individual postings. It also wants to summarize similar postings into broader role categories.

To support that, the notebook normalizes titles by:

- lowercasing them
- removing separators such as `/`, `-`, `_`, `,`, `(`, `)`
- removing noise words such as `senior`, `sr`, `junior`, `jr`, `lead`, `principal`, `staff`, `remote`, `intern`, `contract`, `associate`, `ii`, `iii`, `iv`

This helps reduce duplicates like:

- `Senior Tableau Developer`
- `Tableau Developer`
- `Lead Tableau Developer`

into a more consistent role label.

### 9. Vectorize jobs with TF-IDF

The notebook uses `TfidfVectorizer` from scikit-learn with:

- English stop words removed
- `max_features=10000`
- a token pattern that keeps alphabetic terms

This turns each job posting into a numeric vector. TF-IDF gives more weight to terms that are important within a posting but not overly common across every posting.

The notebook output reports:

- job vector matrix shape: `(111748, 10000)`

This means there are `111,748` job postings represented across `10,000` text features.

### 10. Build student text in the same language space

The student side is converted into text by concatenating:

- `Academic Major`
- `Skill`
- `Area of Experience`
- `School Year`
- `Degree`

This is a simple but meaningful representation of a student's academic and skill profile.

A very important design decision appears next: the student text is transformed with the same TF-IDF vectorizer that was fit on the job postings. That keeps jobs and students in the same feature space, so similarity comparisons are mathematically meaningful.

The notebook output reports:

- student vector matrix shape: `(600, 10000)`

### 11. Retrieve the most similar jobs for a student

The first recommendation layer is retrieval:

- pick one student vector
- compare it to every job vector with cosine similarity
- sort by similarity score
- keep the top `100` postings

This is a good prototype design because it quickly narrows a huge search space to the most relevant postings before doing any higher-level summarization.

In the current notebook, the retrieval example uses:

- `student_vectors[207]`

### 12. Move from posting-level matches to role-level recommendations

If the system stops at individual postings, the top results may contain many near-duplicate listings for essentially the same role. The notebook addresses that by grouping retrieved postings by normalized title.

For each normalized title it computes:

- `avg_similarity`: average similarity across retrieved postings
- `retrieved_posting_count`: how many retrieved postings belong to that title

Then it builds a combined score:

`combined_score = avg_similarity * log1p(retrieved_posting_count)`

This is a useful compromise:

- `avg_similarity` rewards how well the role matches the student
- `log1p(retrieved_posting_count)` rewards repeat demand without letting raw count dominate too aggressively

So the final ranking is not just "what matches best once," but closer to "what matches well and also shows up repeatedly."

### 13. Add location-demand context

Once the top roles are identified, the notebook looks at where those roles appear most often. For each recommended title, it:

- filters the cleaned job dataset to that normalized title
- groups by `state`
- counts postings per state
- shows the top demand states

This gives the student a more useful recommendation than a role name alone. The output becomes something like:

- this role fits your background
- these states currently show the strongest concentration of matching postings

That is the start of a location-aware recommendation system.

### 14. Planned next step: a readiness model

The final markdown section describes a future model that would predict whether a student is best matched to roles such as:

- internship
- entry level
- junior
- advanced or research

That part is not implemented yet in the notebook. Right now it is a design note, not an executed model.

## Recommendation Logic In Plain English

Here is the whole pipeline in one simple flow:

1. Clean the jobs so location and text fields are reliable.
2. Turn each job into a text vector.
3. Turn each student into a text vector using the same vocabulary.
4. Find the jobs whose text looks most similar to the student profile.
5. Group those jobs into broader role titles.
6. Rank the roles by both quality of match and how often they appear.
7. Show where those roles are most common geographically.

So the project is really doing both:

- content-based recommendation, because it compares text content from jobs and students
- market summarization, because it also shows which states have the strongest demand for those matched roles

## What You Need Before Running It

### Required files

You need the following input files:

- [data/uszips.csv](data/uszips.csv), which is already present
- `data/postings.csv`, which is required by the notebook but missing from the repository

If `data/postings.csv` is not present, the notebook will fail at the first job-loading cell.

### Required software

- Python 3.x
- `pip`
- Jupyter Notebook or JupyterLab

The current [requirements.txt](requirements.txt) includes:

- `pandas`
- `numpy`
- `pgeocode`
- `scikit-learn`

Note that the notebook itself does not currently use `pgeocode`, and `jupyter` is not listed in `requirements.txt`, so you should install Jupyter separately.

## Step-By-Step Setup Guide

These commands are written for PowerShell on Windows.

### 1. Open the project folder

```powershell
cd C:\Users\maina\Documents\kdd-project
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install the Python packages

```powershell
pip install -r requirements.txt
pip install notebook
```

If you prefer JupyterLab, you can install that instead:

```powershell
pip install jupyterlab
```

### 6. Add the missing job-posting dataset

Place your job-posting CSV at:

```text
data/postings.csv
```

The notebook expects that file to contain at least these columns:

- `job_id`
- `company_name`
- `title`
- `description`
- `location`
- `company_id`
- `views`
- `skills_desc`
- `work_type`
- `zip_code`
- `fips`

### 7. Confirm internet access for the student dataset

The notebook downloads the student dataset directly from GitHub at runtime. If you are offline, that cell will fail.

If you want to run the project without internet access, download the student CSV yourself and replace the GitHub URL in the notebook with a local file path.

## Step-By-Step Guide To Run The Project

### Option 1: Run with Jupyter Notebook

1. Activate the virtual environment.
2. Start Jupyter:

```powershell
jupyter notebook
```

3. In the browser, open [data_cleaning.ipynb](data_cleaning.ipynb).
4. Run the cells from top to bottom in order.
5. Review the outputs at each section:
   - jobs loaded and cleaned
   - ZIP enrichment
   - student data loaded
   - TF-IDF vector shapes
   - top retrieved postings
   - grouped title recommendations
   - top demand locations by state

### Option 2: Run inside VS Code

1. Open the folder in VS Code.
2. Select the Python interpreter from `.venv`.
3. Open [data_cleaning.ipynb](data_cleaning.ipynb).
4. Run all cells from top to bottom.

### Important: run the notebook in order

The notebook is stateful. Later cells depend on variables created in earlier cells such as:

- `jobs`
- `jobs_text`
- `vectorizer`
- `job_vectors`
- `students_text`
- `student_vectors`
- `title_scores`

If you skip cells or run them out of order, the notebook can break or produce misleading results.

## How To Change The Example Student

The current notebook uses hard-coded student indices for demonstrations:

- retrieval example: `student_vectors[207]`
- location example: `students_text.iloc[97]`

If you want a single student to flow through the entire notebook, define one variable such as:

```python
student_idx = 207
```

and then use that same index in both sections. This will make the recommendation and location examples consistent.

## Expected Outputs

If everything is set up correctly, you should see:

- a cleaned job dataset with resolved `city` and `state`
- a student dataset loaded and normalized
- TF-IDF matrices for jobs and students
- top matched job postings for a selected student
- grouped career-title recommendations
- top demand states for the strongest recommended roles

## Current Limitations

- The entire workflow is in one notebook rather than reusable Python modules or scripts.
- `data/postings.csv` is required but not included in the repository.
- The student dataset depends on a live GitHub URL.
- The notebook examples use different hard-coded student indices in different sections.
- The "Machine Learning Model" section is only a plan and has not been implemented yet.
- The notebook does not currently export recommendations to a file or an API response.

## Suggested Next Improvements

- move reusable logic into Python functions or modules
- define one `student_idx` variable and use it consistently
- save cleaned datasets and recommendation outputs to files
- make dataset paths configurable
- implement the student-readiness classification stage described at the end of the notebook
- add evaluation so recommendation quality can be measured instead of only inspected manually

## Quick Summary

This project is building a content-based student job recommender. It cleans and enriches job data, represents jobs and students as TF-IDF vectors, retrieves similar postings with cosine similarity, aggregates those postings into role-level recommendations, and adds state-level demand information so the results are more actionable.
