#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Generator, List, Optional


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

    def chunk_parameters(self, params: dict, existing_count: int = 0) -> Generator[dict, None, None]:
        """Chunk parameters efficiently using generators, accounting for existing parameters"""
        items = list(params.items())
        # Calculate how many new parameters we can add per chunk
        available_slots = max(1, self.chunk_size - existing_count)

        for i in range(0, len(items), available_slots):
            yield dict(items[i : i + available_slots])

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
        value_strategies: List[str],
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
                        for strategy in value_strategies:
                            new_query = query_params.copy()

                            if strategy == "replace":
                                new_query[param] = [value]
                            elif strategy == "suffix":
                                new_query[param] = [v + value for v in query_params[param]]
                            elif strategy == "prefix":
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
                    existing_count = len(base_query)
                    additional_params = {param: [value] for param in params if param not in base_query}

                    # Skip if no new parameters to add
                    if not additional_params:
                        continue

                    for chunked_additional in self.chunk_parameters(additional_params, existing_count):
                        combined_query = {**base_query, **chunked_additional}
                        new_query = urllib.parse.urlencode(combined_query, doseq=True)
                        new_url = url._replace(query=new_query)
                        yield urllib.parse.urlunparse(new_url)
                except Exception as e:
                    Logger.error(f"URL generation failed in ignore strategy: {e}")

    def generate_urls_to_disk(
        self,
        strategies: List[str],
        urls: List[str],
        values: List[str],
        params: Optional[List[str]] = None,
        value_strategies: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate URLs and write to disk, return list of temp files"""
        processed_urls = self.preprocess_urls(urls)
        if not processed_urls:
            Logger.fatal("No valid URLs to process")

        temp_files = []

        # Generate URLs based on each strategy and write to separate temp files
        for strategy in strategies:
            if strategy == "normal":
                if not params:
                    Logger.fatal("Parameters are required for 'normal' strategy")
                if not self.silent:
                    Logger.info("Starting URL generation with 'normal' strategy")
                generator = self._generate_normal(processed_urls, values, params)
                temp_files.append(self.write_to_temp_file(generator, "normal"))

            elif strategy == "combine":
                if not value_strategies:
                    Logger.fatal("Value strategy is required for 'combine' strategy")
                if not self.silent:
                    Logger.info("Starting URL generation with 'combine' strategy")
                generator = self._generate_combine(processed_urls, values, value_strategies)
                temp_files.append(self.write_to_temp_file(generator, "combine"))

            elif strategy == "ignore":
                if not params:
                    Logger.fatal("Parameters are required for 'ignore' strategy")
                if not self.silent:
                    Logger.info("Starting URL generation with 'ignore' strategy")
                generator = self._generate_ignore(processed_urls, values, params)
                temp_files.append(self.write_to_temp_file(generator, "ignore"))

            elif strategy == "all":
                if not value_strategies:
                    Logger.fatal("Value strategy is required for 'all' strategy")
                if not params:
                    Logger.fatal("Parameters are required for 'all' strategy")
                if not self.silent:
                    Logger.info("Starting URL generation with 'all' strategy (combine + ignore + normal)")

                # Generate each strategy to separate file
                generator = self._generate_combine(processed_urls, values, value_strategies)
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
    if args.values:
        print(f"Values Source   : Inline arguments", file=sys.stderr)
    else:
        print(f"Values Source   : File ({args.values_file})", file=sys.stderr)

    print(f"Values Count    : {values_count}", file=sys.stderr)

    # Parameters
    if args.parameters:
        print(f"Parameters      : Inline arguments", file=sys.stderr)
        print(f"Parameters Count: {params_count}", file=sys.stderr)
    elif args.parameters_file:
        print(f"Parameters      : File ({args.parameters_file})", file=sys.stderr)
        print(f"Parameters Count: {params_count}", file=sys.stderr)
    else:
        print(f"Parameters      : None", file=sys.stderr)

    # Strategy
    print(f"Generate Strategy: {', '.join(args.generate_strategy)}", file=sys.stderr)

    if args.value_strategy:
        print(f"Value Strategy   : {', '.join(args.value_strategy)}", file=sys.stderr)

    print(f"Chunk Size       : {args.chunk}", file=sys.stderr)

    # Output
    if output_file:
        print(f"Output           : {output_file}", file=sys.stderr)
    else:
        print(f"Output           : stdout", file=sys.stderr)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", file=sys.stderr)


def parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated values into a list"""
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(
        description="Param Ninja - Advanced URL Parameter Fuzzing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy Descriptions:
  normal   : Generate new URLs with only the specified parameters and values
  combine  : Inject values into existing URL parameters
  ignore   : Add new parameters to existing URLs, ignoring duplicates
  all      : Execute all three strategies (normal, combine, ignore)

Value Strategy Descriptions:
  replace  : Replace parameter values entirely with the injection value
  suffix   : Append injection value to the end of existing parameter values
  prefix   : Prepend injection value to the beginning of existing parameter values
  all      : Apply all three value strategies (replace, suffix, prefix)

Notes:
  - Multiple strategies can be specified using comma separation (e.g., normal,combine)
  - Multiple value strategies can be specified using comma separation (e.g., replace,suffix)
  - Values containing special characters should be enclosed in quotes
  - Chunk size includes existing parameters in the URL
""",
    )

    # ============================================================================
    # INPUT SOURCE
    # ============================================================================
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument("-u", "--url", metavar="URL", help="Single target URL to fuzz")
    url_group.add_argument("-U", "--url-list", metavar="FILE", dest="url_list", help="File containing list of URLs (one per line)")

    # ============================================================================
    # GENERATION STRATEGY
    # ============================================================================
    parser.add_argument(
        "-gs",
        "--generate-strategy",
        metavar="STRATEGY",
        dest="generate_strategy",
        required=True,
        type=parse_comma_separated,
        help="URL generation strategy: all, combine, normal, ignore (comma-separated for multiple)",
    )

    # ============================================================================
    # VALUE STRATEGY
    # ============================================================================
    parser.add_argument(
        "-vs",
        "--value-strategy",
        metavar="STRATEGY",
        dest="value_strategy",
        type=parse_comma_separated,
        help="Value injection strategy: replace, suffix, prefix, all (comma-separated for multiple). Required for 'combine' and 'all' strategies",
    )

    # ============================================================================
    # VALUES INPUT
    # ============================================================================
    value_group = parser.add_mutually_exclusive_group(required=True)
    value_group.add_argument("-v", "--values", metavar="VALUE", nargs="+", help="Injection values provided as inline arguments (quote values with special characters)")
    value_group.add_argument("-V", "--values-file", metavar="FILE", dest="values_file", help="File containing injection values (one per line)")

    # ============================================================================
    # PARAMETERS INPUT
    # ============================================================================
    param_group = parser.add_mutually_exclusive_group()
    param_group.add_argument(
        "-p",
        "--parameters",
        metavar="PARAM",
        type=parse_comma_separated,
        help="Parameter names provided as inline arguments (comma-separated). Required for 'normal', 'ignore', and 'all' strategies",
    )
    param_group.add_argument(
        "-P",
        "--parameters-file",
        metavar="FILE",
        dest="parameters_file",
        help="File containing parameter names (one per line). Required for 'normal', 'ignore', and 'all' strategies",
    )

    # ============================================================================
    # OPTIONS
    # ============================================================================
    parser.add_argument(
        "-c", "--chunk", metavar="N", type=int, default=25, help="Maximum number of parameters per generated URL (default: 25). Includes existing parameters in the URL"
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        nargs="?",
        const="",
        default=None,
        help="Output file path. If flag provided without filename, auto-generates based on input. If omitted, outputs to stdout",
    )
    parser.add_argument("-s", "--silent", action="store_true", help="Silent mode - suppress banner and progress messages")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files after execution for debugging purposes")

    args = parser.parse_args()

    # ============================================================================
    # PARAMETER VALIDATION
    # ============================================================================

    # Validate generate_strategy values
    valid_gen_strategies = {"all", "combine", "normal", "ignore"}
    for strategy in args.generate_strategy:
        if strategy not in valid_gen_strategies:
            Logger.fatal(f"Invalid generate strategy: '{strategy}'. Must be one of: {', '.join(valid_gen_strategies)}")

    # Validate value_strategy values if provided
    if args.value_strategy:
        valid_val_strategies = {"replace", "suffix", "prefix", "all"}
        for strategy in args.value_strategy:
            if strategy not in valid_val_strategies:
                Logger.fatal(f"Invalid value strategy: '{strategy}'. Must be one of: {', '.join(valid_val_strategies)}")

        # Expand 'all' to all value strategies
        if "all" in args.value_strategy:
            args.value_strategy = ["replace", "suffix", "prefix"]

    # Check if parameters are required
    requires_params = any(s in args.generate_strategy for s in ["ignore", "normal", "all"])
    if requires_params and not args.parameters and not args.parameters_file:
        Logger.fatal(f"Parameters (-p or -P) are required for strategies: {', '.join([s for s in args.generate_strategy if s in ['ignore', 'normal', 'all']])}")

    # Check if value strategy is required
    requires_value_strategy = any(s in args.generate_strategy for s in ["combine", "all"])
    if requires_value_strategy and not args.value_strategy:
        Logger.fatal(f"Value strategy (-vs) is required for strategies: {', '.join([s for s in args.generate_strategy if s in ['combine', 'all']])}")

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
        if args.values:
            values = args.values
        else:
            values = fuzzer.load_file(args.values_file)

        # Get parameters from inline arguments or file
        params = None
        params_count = 0
        if args.parameters:
            params = args.parameters
            params_count = len(params)
        elif args.parameters_file:
            params = fuzzer.load_file(args.parameters_file)
            params_count = len(params)

        # Determine output file
        output_file = generate_output_filename(args)

        # Print configuration
        print_configuration(args, len(links), len(values), params_count, output_file, args.silent)

        # Generate URLs to temporary files
        if not args.silent:
            Logger.info("Starting URL generation process...")

        temp_files = fuzzer.generate_urls_to_disk(
            strategies=args.generate_strategy,
            urls=links,
            values=values,
            params=params,
            value_strategies=args.value_strategy,
        )

        if not temp_files:
            Logger.fatal("No URLs generated")

        # Sort and deduplicate each temp file
        if not args.silent:
            Logger.info("Starting deduplication process...")

        sorted_files = []

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
