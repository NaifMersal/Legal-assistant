# Legal Assistant UI

## Project Setup

### Prerequisites

Ensure you have Node.js and npm installed. You can use [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) to manage Node.js versions.

### Steps to Run Locally

1. Clone the repository:
   ```sh
   git clone <YOUR_GIT_URL>
   ```

2. Navigate to the project directory:
   ```sh
   cd <YOUR_PROJECT_NAME>
   ```

3. Install dependencies:
   ```sh
   npm install
   ```

4. Start the development server:
   ```sh
   npm run dev
   ```

The development server will start with auto-reloading and an instant preview.

## Technologies Used

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Deployment

To deploy the project, follow your preferred deployment strategy for Vite-based React applications.

### Integration with Backend

This UI is designed to work with the Legal Assistant backend. Ensure the backend server is running and accessible. Update the API endpoint in `src/lib/api.ts` if necessary.
