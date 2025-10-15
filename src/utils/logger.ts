/**
 * Debug Logger
 *
 * Provides debug logging to /tmp/memory-mcp/<instance-id>.log
 * Only active when debug mode is enabled.
 * Logs are written as JSON lines for easy parsing.
 */

import * as fs from 'fs/promises';
import * as path from 'path';

const LOG_DIR = '/tmp/memory-mcp';

/**
 * Debug logger interface
 */
export interface Logger {
  debug(operation: string, data: Record<string, unknown>): Promise<void>;
  close(): Promise<void>;
}

/**
 * No-op logger for when debug mode is disabled
 */
class NoOpLogger implements Logger {
  async debug(_operation: string, _data: Record<string, unknown>): Promise<void> {
    // Do nothing
  }

  async close(): Promise<void> {
    // Do nothing
  }
}

/**
 * File-based debug logger
 */
class FileLogger implements Logger {
  private logFile: string;
  private writeQueue: Promise<void> = Promise.resolve();

  constructor(instanceId: string) {
    this.logFile = path.join(LOG_DIR, `${instanceId}.log`);
  }

  /**
   * Initialize logger by creating log directory and file
   */
  async init(): Promise<void> {
    // Ensure log directory exists
    await fs.mkdir(LOG_DIR, { recursive: true });

    // Create/truncate log file with initial entry
    const initEntry = {
      timestamp: new Date().toISOString(),
      level: 'info',
      message: 'Debug logging started',
      instanceId: path.basename(this.logFile, '.log'),
    };

    await fs.writeFile(this.logFile, JSON.stringify(initEntry) + '\n', 'utf-8');
  }

  /**
   * Log a debug entry
   * Entries are queued to ensure sequential writes
   */
  async debug(operation: string, data: Record<string, unknown>): Promise<void> {
    // Queue write to ensure sequential ordering
    this.writeQueue = this.writeQueue.then(async () => {
      // Build log entry
      const entry = {
        timestamp: new Date().toISOString(),
        level: 'debug',
        operation,
        ...data,
      };

      // Write to log file
      try {
        await fs.appendFile(this.logFile, JSON.stringify(entry) + '\n', 'utf-8');
      } catch (error) {
        // Fail silently - don't let logging errors crash the server
        console.error('Failed to write to debug log:', error);
      }
    });

    return this.writeQueue;
  }

  /**
   * Close logger and flush any pending writes
   */
  async close(): Promise<void> {
    // Wait for all queued writes to complete
    await this.writeQueue;
  }
}

/**
 * Generate a unique instance ID for this server instance
 *
 * Format: <timestamp>-<random>
 * Example: 20251015-143022-a3f9
 */
function generateInstanceId(): string {
  // Format timestamp as YYYYMMDD-HHMMSS
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  const timestamp = `${year}${month}${day}-${hours}${minutes}${seconds}`;

  // Add random suffix for uniqueness
  const random = Math.random().toString(36).substring(2, 6);

  return `${timestamp}-${random}`;
}

/**
 * Create a logger instance
 *
 * @param debug - Whether debug logging is enabled
 * @returns Logger instance (file-based if debug=true, no-op otherwise)
 */
export async function createLogger(debug: boolean): Promise<Logger> {
  if (!debug) {
    return new NoOpLogger();
  }

  // Create file logger
  const instanceId = generateInstanceId();
  const logger = new FileLogger(instanceId);

  // Initialize log file
  await logger.init();

  return logger;
}
