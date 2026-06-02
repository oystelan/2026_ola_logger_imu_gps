/**
 * @file file_transfer.h
 * @brief USB-serial file transfer mode for the OLA.
 *
 * Entered from main.cpp when the RESET multi-press detector returns
 * BootAction::FILE_TRANSFER (2 presses within the decision window). The
 * device skips all IMU / GNSS / logging setup, brings up the SD card, and
 * loops on SERIAL_USB serving a small text-based command protocol so a
 * host PC can list and download recordings without removing the SD card.
 *
 * Protocol (one ASCII line per command, '\n' or '\r\n' terminated):
 *
 *   list                Enumerate every *.dat file under BOOT_NNNNNN/.
 *                       Response: LIST_BEGIN, lines of "<idx> <bytes> <path>",
 *                       LIST_END <count>.
 *
 *   get <N>             Stream file at index N (from `list`) as a length-
 *                       prefixed binary blob followed by a CRC32 trailer.
 *                       Response: "GET_BEGIN <N> <bytes>\n" then exactly
 *                       <bytes> raw binary, then "\nGET_END <crc32-hex>".
 *
 *   info                Print firmware commit/branch and SD card info.
 *
 *   help                Print the command list.
 *
 *   exit | reboot       Leave transfer mode and reboot into normal logging.
 *
 *   <anything else>     ERROR unknown_command
 *
 * Every command (incl. errors) ends with a trailing "READY" line so the
 * host can synchronise without races.
 *
 * The function never returns — exit triggers NVIC_SystemReset(). The host
 * is expected to wait for "READY" once after the boot banner before
 * sending the first command.
 */
#ifndef FILE_TRANSFER_H
#define FILE_TRANSFER_H

void enter_file_transfer_mode(void);

#endif
