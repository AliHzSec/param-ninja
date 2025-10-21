#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import urllib.parse
import tempfile
import subprocess
import os
from typing import List, Optional, Generator


class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"


class Logger:
    """Logging utility that writes to stderr with colored prefixes"""

    @staticmethod
    def info(message: str):
        """Print info message to stderr"""
        prefix = f"{Colors.BLUE}[INF]{Colors.RESET}"
        print(f"{prefix} {message}", file=sys.stderr)

    @staticmethod
    def success(message: str):
        """Print success message to stderr"""
        prefix = f"{Colors.GREEN}[SUC]{Colors.RESET}"
        print(f"{prefix} {message}", file=sys.stderr)

    @staticmethod
    def warning(message: str):
        """Print warning message to stderr"""
        prefix = f"{Colors.YELLOW}[WRN]{Colors.RESET}"
        print(f"{prefix} {message}", file=sys.stderr)

    @staticmethod
    def error(message: str):
        """Print error message to stderr"""
        prefix = f"{Colors.RED}[ERR]{Colors.RESET}"
        print(f"{prefix} {message}", file=sys.stderr)

    @staticmethod
    def fatal(message: str):
        """Print fatal error message to stderr and exit"""
        prefix = f"{Colors.RED}[FTL]{Colors.RESET}"
        print(f"{prefix} {message}", file=sys.stderr)
        sys.exit(1)


class URLFuzzer:
    """Memory-efficient URL fuzzer with disk-based deduplication"""

    def __init__(self, chunk_size: int = 25, silent: bool = False):
        """Initialize the URLFuzzer with chunk size"""
        self.chunk_size = chunk_size
        self.silent = silent
        self.temp_files = []

    @staticmethod
    def print_banner() -> None:
        """Print the tool's banner"""
        banner = r"""
     ____                                    _   __ _           _       
    / __ \ ____ _ _____ ____ _ ____ ___     / | / /(_)____     (_)____ _
   / /_/ // __ `// ___// __ `// __ `__ \   /  |/ // // __ \   / // __ `/
  / ____// /_/ // /   / /_/ // / / / / /  / /|  // // / / /  / // /_/ / 
 /_/     \__,_//_/    \__,_//_/ /_/ /_/  /_/ |_//_//_/ /_/__/ / \__,_/  
                                                         /___/          

"""
        print(banner, file=sys.stderr)

    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and decode a URL"""
        try:
            url = urllib.parse.unquote(url)
            url = url.replace("\\", "")
            return url
        except Exception as e:
            Logger.error(f"URL cleaning failed: {e}")
            return url

    @staticmethod
    def load_file(filename: str) -> List[str]:
        """Load data from a file with error handling"""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return [line.strip() for line in file.readlines() if line.strip()]
        except (IOError, FileNotFoundError) as e:
            Logger.fatal(f"Failed to load file '{filename}': {e}")
        except Exception as e:
            Logger.fatal(f"Unexpected error loading file '{filename}': {e}")

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if a string is a proper URL"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def preprocess_urls(self, urls: List[str]) -> List[urllib.parse.ParseResult]:
        """Preprocess all URLs once to avoid redundant operations"""
        processed_urls = []
        invalid_count = 0

        for url in urls:
            clean_url = self.clean_url(url)
            if not self.validate_url(clean_url):
                invalid_count += 1
                continue

            try:
                processed_urls.append(urllib.parse.urlparse(clean_url))
            except Exception as e:
                Logger.warning(f"URL parsing failed: {e}")
                invalid_count += 1

        if invalid_count > 0 and not self.silent:
            Logger.warning(f"Skipped {invalid_count} invalid URLs")

        return processed_urls

    def write_to_temp_file(self, generator: Generator, strategy_name: str) -> str:
        """Write generator output to a temporary file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, prefix=f"x9_{strategy_name}_", suffix=".txt", encoding="utf-8")
        self.temp_files.append(temp_file.name)

        count = 0
        try:
            for url in generator:
                temp_file.write(f"{url}\n")
                count += 1

                if count % 1000 == 0 and not self.silent:
                    Logger.info(f"[{strategy_name}] Generated {count} URLs...")

            temp_file.close()

            if not self.silent:
                Logger.success(f"[{strategy_name}] Total {count} URLs written")

            return temp_file.name

        except Exception as e:
            Logger.error(f"Failed to write temp file for {strategy_name}: {e}")
            temp_file.close()
            return temp_file.name

    def sort_and_deduplicate_file(self, input_file: str, strategy_name: str) -> str:
        """Sort and deduplicate a file using Unix sort command"""
        output_file = f"{input_file}.sorted"

        try:
            if not self.silent:
                Logger.info(f"[{strategy_name}] Sorting and deduplicating...")

            with open(output_file, "w", encoding="utf-8") as outf:
                subprocess.run(["sort", "-u", input_file], stdout=outf, stderr=subprocess.DEVNULL, check=True)

            # Get line counts for stats
            original_count = int(subprocess.check_output(["wc", "-l", input_file]).split()[0])
            sorted_count = int(subprocess.check_output(["wc", "-l", output_file]).split()[0])
            duplicates = original_count - sorted_count

            if not self.silent:
                Logger.success(f"[{strategy_name}] {original_count} → {sorted_count} unique (removed {duplicates} duplicates)")

            return output_file

        except subprocess.CalledProcessError as e:
            Logger.error(f"Sort command failed for {strategy_name}: {e}")
            return input_file
        except Exception as e:
            Logger.error(f"Unexpected error during sorting: {e}")
            return input_file

    def merge_sorted_files(self, sorted_files: List[str], output_file: Optional[str] = None) -> int:
        """Merge multiple sorted files and remove duplicates. Returns count of unique URLs."""
        try:
            if not self.silent:
                Logger.info(f"Merging {len(sorted_files)} sorted files...")

            if output_file:
                # Write to file (no stdout output)
                with open(output_file, "w", encoding="utf-8") as outf:
                    subprocess.run(["sort", "-mu"] + sorted_files, stdout=outf, stderr=subprocess.DEVNULL, check=True)

                final_count = int(subprocess.check_output(["wc", "-l", output_file]).split()[0])
                Logger.success(f"Final output: {final_count} unique URLs written to {output_file}")
                return final_count
            else:
                # Write to stdout (no file output)
                subprocess.run(["sort", "-mu"] + sorted_files, stderr=subprocess.DEVNULL, check=True)

                if not self.silent:
                    Logger.info("Output written to stdout")
                return 0

        except subprocess.CalledProcessError as e:
            Logger.fatal(f"Failed to merge files: {e}")
        except Exception as e:
            Logger.fatal(f"Unexpected error during merge: {e}")

    def cleanup_temp_files(self):
        """Clean up all temporary files"""
        if not self.silent:
            Logger.info("Cleaning up temporary files...")

        cleaned = 0
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned += 1
                sorted_file = f"{temp_file}.sorted"
                if os.path.exists(sorted_file):
                    os.remove(sorted_file)
                    cleaned += 1
            except Exception as e:
                Logger.warning(f"Failed to delete temp file: {e}")

        if not self.silent and cleaned > 0:
            Logger.success(f"Removed {cleaned} temporary files")

    def chunk_parameters(self, params: dict) -> Generator[dict, None, None]:
        """Chunk parameters efficiently using generators"""
        items = list(params.items())
        for i in range(0, len(items), self.chunk_size):
            yield dict(items[i : i + self.chunk_size])

    def _generate_normal(
        self,
        processed_urls: List[urllib.parse.ParseResult],
        values: List[str],
        params: List[str],
    ) -> Generator[str, None, None]:
        """Generate URLs based on 'normal' strategy"""
        for value in values:
            for url in processed_urls:
                new_params = {param: [value] for param in params}

                for chunked_params in self.chunk_parameters(new_params):
                    try:
                        new_query = urllib.parse.urlencode(chunked_params, doseq=True)
                        new_url = url._replace(query=new_query)
                        yield urllib.parse.urlunparse(new_url)
                    except Exception as e:
                        Logger.error(f"URL generation failed in normal strategy: {e}")

    def _generate_combine(
        self,
        processed_urls: List[urllib.parse.ParseResult],
        values: List[str],
        value_strategy: str,
    ) -> Generator[str, None, None]:
        """Generate URLs based on 'combine' strategy"""
        for url in processed_urls:
            try:
                query_params = urllib.parse.parse_qs(url.query)
                if not query_params:
                    continue

                base_url = url._replace(query="")

                for value in values:
                    for param in query_params.keys():
                        new_query = query_params.copy()

                        if value_strategy == "replace":
                            new_query[param] = [value]
                        elif value_strategy == "suffix":
                            new_query[param] = [v + value for v in query_params[param]]
                            query_string = urllib.parse.urlencode(new_query, doseq=True)
                            new_url = base_url._replace(query=query_string)
                            yield urllib.parse.urlunparse(new_url)

                            new_query[param] = [value + v for v in query_params[param]]

                        query_string = urllib.parse.urlencode(new_query, doseq=True)
                        new_url = base_url._replace(query=query_string)
                        yield urllib.parse.urlunparse(new_url)
            except Exception as e:
                Logger.error(f"URL generation failed in combine strategy: {e}")

    def _generate_ignore(
        self,
        processed_urls: List[urllib.parse.ParseResult],
        values: List[str],
        params: List[str],
    ) -> Generator[str, None, None]:
        """Generate URLs based on 'ignore' strategy"""
        for value in values:
            for url in processed_urls:
                try:
                    base_query = urllib.parse.parse_qs(url.query)
                    additional_params = {param: [value] for param in params if param not in base_query}

                    # Skip if no new parameters to add
                    if not additional_params:
                        continue

                    for chunked_additional in self.chunk_parameters(additional_params):
                        combined_query = {**base_query, **chunked_additional}
                        new_query = urllib.parse.urlencode(combined_query, doseq=True)
                        new_url = url._replace(query=new_query)
                        yield urllib.parse.urlunparse(new_url)
                except Exception as e:
                    Logger.error(f"URL generation failed in ignore strategy: {e}")

    def generate_urls_to_disk(
        self,
        strategy: str,
        urls: List[str],
        values: List[str],
        params_file: Optional[str] = None,
        value_strategy: Optional[str] = None,
    ) -> List[str]:
        """Generate URLs and write to disk, return list of temp files"""
        processed_urls = self.preprocess_urls(urls)
        if not processed_urls:
            Logger.fatal("No valid URLs to process")

        # Load parameters if needed
        params = []
        if params_file and strategy in ["ignore", "normal", "all"]:
            params = self.load_file(params_file)
            if not self.silent:
                Logger.info(f"Loaded {len(params)} parameters from file")

        temp_files = []

        # Generate URLs based on strategy and write to separate temp files
        if strategy == "normal":
            if not self.silent:
                Logger.info("Starting URL generation with 'normal' strategy")
            generator = self._generate_normal(processed_urls, values, params)
            temp_files.append(self.write_to_temp_file(generator, "normal"))

        elif strategy == "combine":
            if not value_strategy:
                Logger.fatal("Value strategy is required for 'combine' strategy")

            if not self.silent:
                Logger.info("Starting URL generation with 'combine' strategy")
            generator = self._generate_combine(processed_urls, values, value_strategy)
            temp_files.append(self.write_to_temp_file(generator, "combine"))

        elif strategy == "ignore":
            if not self.silent:
                Logger.info("Starting URL generation with 'ignore' strategy")
            generator = self._generate_ignore(processed_urls, values, params)
            temp_files.append(self.write_to_temp_file(generator, "ignore"))

        elif strategy == "all":
            if not value_strategy:
                Logger.fatal("Value strategy is required for 'all' strategy")

            if not self.silent:
                Logger.info("Starting URL generation with 'all' strategy (combine + ignore + normal)")

            # Generate each strategy to separate file
            generator = self._generate_combine(processed_urls, values, value_strategy)
            temp_files.append(self.write_to_temp_file(generator, "combine"))

            generator = self._generate_ignore(processed_urls, values, params)
            temp_files.append(self.write_to_temp_file(generator, "ignore"))

            generator = self._generate_normal(processed_urls, values, params)
            temp_files.append(self.write_to_temp_file(generator, "normal"))

        return temp_files


def generate_output_filename(args) -> Optional[str]:
    """Generate output filename based on input source"""
    # If -o not provided at all, return None (stdout)
    if args.output is None:
        return None

    # If -o provided with a filename, use it
    if args.output:
        return args.output

    # If -o provided without filename, auto-generate
    if args.url:
        # Extract domain from URL
        try:
            parsed = urllib.parse.urlparse(args.url)
            domain = parsed.netloc if parsed.netloc else "output"
            return f"{domain}_x9-generated.txt"
        except:
            return "output_x9-generated.txt"
    else:
        # Use input filename + suffix
        return f"{args.url_list}_x9-generated.txt"


def print_configuration(args, links_count: int, values_count: int, params_count: int, output_file: Optional[str], silent: bool):
    """Print configuration summary"""
    if silent:
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", file=sys.stderr)

    # Input source
    if args.url:
        print(f"Input Source    : Single URL", file=sys.stderr)
    else:
        print(f"Input Source    : File ({args.url_list})", file=sys.stderr)

    print(f"Input URLs      : {links_count}", file=sys.stderr)

    # Values source
    if args.values_inline:
        print(f"Values Source   : Inline arguments", file=sys.stderr)
    else:
        print(f"Values Source   : File ({args.values_file})", file=sys.stderr)

    print(f"Values Count    : {values_count}", file=sys.stderr)

    # Parameters
    if args.parameters:
        print(f"Parameters File : {args.parameters}", file=sys.stderr)
        print(f"Parameters Count: {params_count}", file=sys.stderr)
    else:
        print(f"Parameters File : None", file=sys.stderr)

    # Strategy
    print(f"Strategy        : {args.generate_strategy}", file=sys.stderr)

    if args.value_strategy:
        print(f"Value Strategy  : {args.value_strategy}", file=sys.stderr)

    print(f"Chunk Size      : {args.chunk}", file=sys.stderr)

    # Output
    if output_file:
        print(f"Output          : {output_file}", file=sys.stderr)
    else:
        print(f"Output          : stdout", file=sys.stderr)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", file=sys.stderr)


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input URL/URLs
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument("-u", "--url", help="Single URL")
    url_group.add_argument("-l", "--url_list", help="File containing URLs (one per line)")

    # Strategy
    parser.add_argument(
        "-gs",
        "--generate_strategy",
        required=True,
        choices=["all", "combine", "normal", "ignore"],
        help="URL generation strategy",
    )
    parser.add_argument(
        "-vs",
        "--value_strategy",
        choices=["replace", "suffix"],
        help="Value application strategy (required for 'all' and 'combine')",
    )

    # Values
    value_group = parser.add_mutually_exclusive_group(required=True)
    value_group.add_argument("-v", "--values_inline", nargs="+", help="Values provided as inline arguments")
    value_group.add_argument("-vf", "--values_file", help="File containing values (one per line)")

    # Parameters
    parser.add_argument(
        "-p",
        "--parameters",
        help="File containing parameters (required for 'ignore', 'normal', and 'all' strategies)",
    )

    # Options
    parser.add_argument("-c", "--chunk", type=int, default=25, help="Number of parameters per URL chunk")
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="",
        default=None,
        help="Output file (if no filename provided, auto-generates based on input)",
    )
    parser.add_argument("-s", "--silent", action="store_true", help="Silent mode - suppress banner and progress messages")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files for debugging")

    args = parser.parse_args()

    # Parameter validation
    if args.generate_strategy in ["ignore", "normal", "all"] and not args.parameters:
        Logger.fatal(f"Parameter file (-p) is required for '{args.generate_strategy}' strategy")

    if args.generate_strategy in ["combine", "all"] and not args.value_strategy:
        Logger.fatal(f"Value strategy (-vs) is required for '{args.generate_strategy}' strategy")

    # Print banner if not in silent mode
    if not args.silent:
        URLFuzzer.print_banner()

    # Set up fuzzer
    fuzzer = URLFuzzer(chunk_size=args.chunk, silent=args.silent)

    try:
        # Get links from URL or file
        if args.url:
            links = [args.url]
        else:
            links = fuzzer.load_file(args.url_list)

        # Get values from inline arguments or file
        if args.values_inline:
            values = args.values_inline
        else:
            values = fuzzer.load_file(args.values_file)

        # Get parameters count if file provided
        params_count = 0
        if args.parameters:
            params = fuzzer.load_file(args.parameters)
            params_count = len(params)

        # Determine output file
        output_file = generate_output_filename(args)

        # Print configuration
        print_configuration(args, len(links), len(values), params_count, output_file, args.silent)

        # Generate URLs to temporary files
        if not args.silent:
            Logger.info("Starting URL generation process...")

        temp_files = fuzzer.generate_urls_to_disk(
            strategy=args.generate_strategy,
            urls=links,
            values=values,
            params_file=args.parameters,
            value_strategy=args.value_strategy,
        )

        if not temp_files:
            Logger.fatal("No URLs generated")

        # Sort and deduplicate each temp file
        if not args.silent:
            Logger.info("Starting deduplication process...")

        sorted_files = []
        strategy_names = {
            "normal": "normal",
            "combine": "combine",
            "ignore": "ignore",
        }

        for i, temp_file in enumerate(temp_files):
            # Determine strategy name from temp file
            if "combine" in temp_file:
                strategy_name = "combine"
            elif "ignore" in temp_file:
                strategy_name = "ignore"
            elif "normal" in temp_file:
                strategy_name = "normal"
            else:
                strategy_name = f"strategy_{i+1}"

            sorted_file = fuzzer.sort_and_deduplicate_file(temp_file, strategy_name)
            sorted_files.append(sorted_file)

        # Merge all sorted files
        if not args.silent:
            Logger.info("Starting merge process...")

        fuzzer.merge_sorted_files(sorted_files, output_file)

        if not args.silent:
            Logger.success("URL fuzzing completed successfully!")

    except KeyboardInterrupt:
        Logger.warning("Process interrupted by user (Ctrl-C)")
        sys.exit(1)
    except Exception as e:
        Logger.fatal(f"Unexpected error: {e}")
    finally:
        # Cleanup temp files unless --keep-temp is specified
        if not args.keep_temp:
            fuzzer.cleanup_temp_files()
        else:
            if not args.silent:
                Logger.info(f"Temporary files kept in: {', '.join(fuzzer.temp_files)}")


if __name__ == "__main__":
    main()
