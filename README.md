# AJSA (Automated Job Search Assistant)

## Overview
AJSA was born from a personal journey - watching my wife navigate the challenging waters of job searching. It's more than just another job application tool; it's a testament to how technology can make career transitions more manageable and human-centered.

This application streamlines the job application process by:
- Automatically tracking and organizing job applications
- Providing smart resume tailoring suggestions
- Managing application deadlines and follow-ups
- Offering insights into application status and progress

Built with love and empathy, AJSA aims to reduce the stress and uncertainty of job searching, helping job seekers focus on what truly matters - finding their next great opportunity.

### Why AJSA?
- **Personal Touch**: Created with real job seekers in mind
- **Time-Saving**: Automates repetitive tasks in the application process
- **Organization**: Keep all your applications, follow-ups, and documents in one place
- **Peace of Mind**: Never miss an application deadline or follow-up again

Remember: Every great career journey starts with a single application. Let AJSA be your companion in this journey.


## Prerequisites
- Docker installed on your system
- `.env` file with required environment variables
- SQLite database file (`ajsa_database.sqlite3`)

## Quick Start

### Step 1: Environment Setup
1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and configure your `.env` file:
```bash
cp .env.example .env
# Edit .env with your settings
```
3. Initialize the database:
```bash
python utils/init_db.py
```

This will create a SQLite database (`ajsa_database.sqlite3`) with the following structure:

#### Database Schema
- **applications**
  - Stores job application details (job title, company, resume, cover letter, etc.)
  - Primary key: `id` (UUID)
  - Tracks application dates and status

- **current_application**
  - Manages the currently active application
  - Links to applications table via foreign key
  - Ensures single active application workflow

- **analysis_steps**
  - Records analysis steps for each application with the following workflow:
    1. **Job Posting Analysis**
       - Extracts key requirements and skills
       - Identifies company values and culture
       - Determines seniority level and experience needs

    2. **Resume Tailoring**
       - Matches candidate skills with job requirements
       - Suggests relevant experience highlights
       - Recommends keyword optimizations

    3. **Cover Letter Generation**
       - Creates personalized introduction
       - Aligns experience with job requirements
       - Emphasizes cultural fit
       - Concludes with call to action

    4. **Application Review** *(Coming Soon)* 🚧
       - Performs final compatibility check
       - Validates document formatting
       - Ensures all requirements are addressed

    5. **Follow-up Tracking** *(Coming Soon)* 🚧
       - Records application submission date
       - Sets reminder for follow-up communications
       - Tracks interview stages and feedback

Each step is timestamped and linked to the specific application via foreign key.

*Note: Features marked with 🚧 are planned for future releases.*

Each step is timestamped and linked to the specific application via foreign key

### Step 2: Build the Docker Image
To build the Docker image, run the following command in your terminal:

```bash
docker build -t streamlit-app .
```

### Step 3: Create and Run a Container
After building the image, create and run a container:

```bash
docker run -d \
    --name ajsa-app \
    --restart unless-stopped \
    -p 8501:8501 \
    -v $(pwd)/ajsa_database.sqlite3:/app/ajsa_database.sqlite3 \
    -v $(pwd)/.env:/app/.env \
    streamlit-app
```

**⚠️ Warning:** Ensure that the `ajsa_database.sqlite3` and `.env` files exist in the current directory before running the container.

### Step 4: Verify the Container
```bash
# Check container status
docker ps

# View container logs
docker logs ajsa-app

# Check the volume mount
docker inspect ajsa-app | grep -A 20 Mounts
```

## Configuration
Describe important configuration options and environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Application port | 8510 |
| ... | ... | ... |

## Development

### Local Development Setup
Instructions for setting up the development environment locally.

### Testing
Instructions for running tests.

## Troubleshooting

### Common Issues
1. **Container fails to start**
   - Check if all required files exist
   - Verify port availability
   - Check logs using `docker logs ajsa-app`

2. **Database connection issues**
   - Verify database file permissions
   - Check database file path

## Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull

# Rebuild container
docker build -t streamlit-app .

# Stop and remove old container
docker stop ajsa-app
docker rm ajsa-app

# Start new container
docker run -d [... previous run command ...]
```

### Backup and Restore
Instructions for backing up and restoring data.

## Contributing
Guidelines for contributing to the project.

## License
Specify your project's license.

## Contact
How to reach the maintainers.