/**
 * Error handling utilities
 *
 * Provides type-safe helpers for extracting information from unknown error types.
 */

/**
 * Safely extract error code from unknown error
 *
 * @param error - Unknown error object
 * @returns Error code if available, undefined otherwise
 */
export function getErrorCode(error: unknown): string | undefined {
  if (error && typeof error === 'object' && 'code' in error && typeof error.code === 'string') {
    return error.code;
  }
  return undefined;
}

/**
 * Safely extract error message from unknown error
 *
 * @param error - Unknown error object
 * @returns Error message as string
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
