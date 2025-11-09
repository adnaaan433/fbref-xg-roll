from altair import FontWeight
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from highlight_text import fig_text
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils.driver import get_driver

st.set_page_config(page_title="FBref Player Chart", page_icon="⚽", layout="wide")

st.title("⚽ Players' Line Chart")
st.markdown("Enter a player's FBref URL to view their logs")


@st.cache_data(show_spinner=False)
def fetch_goal_logs(url: str) -> pd.DataFrame:
    """
    Fetch goal logs from FBref URL and return as DataFrame.
    Results are cached based on URL.

    Args:
        url: The FBref goal logs URL

    Returns:
        DataFrame containing goal logs

    Raises:
        ValueError: If no table is found
        Exception: For other errors during fetching
    """
    with get_driver() as driver:
        driver.get(url)

        # Wait a moment for page to load
        time.sleep(1)

        # Get page source
        page_source = driver.page_source

    # Parse with BeautifulSoup
    soup = BeautifulSoup(page_source, "lxml")

    # Find table with id 'goallogs_goals'
    table = soup.find("table", {"id": "goallogs_goals"})

    if table:
        # Convert table to pandas DataFrame
        goal_logs_df = pd.read_html(str(table))[0]
        return goal_logs_df
    else:
        raise ValueError("No goal logs table found on the page")


@st.cache_data(show_spinner=False)
def fetch_match_logs(url: str, remove_national_team: bool = False) -> pd.DataFrame:
    """
    Fetch match logs from FBref URL and return as DataFrame.
    Results are cached based on URL and remove_national_team parameter.

    Args:
        url: The FBref player URL
        remove_national_team: If True, removes rows where Squad or Opponent matches the national team

    Returns:
        DataFrame containing all match logs from all seasons (excluding national team competition pages)

    Raises:
        ValueError: If no match logs are found
        Exception: For other errors during fetching
    """
    with get_driver() as driver:
        driver.get(url)

        # Wait a moment for page to load
        time.sleep(1)

        # Get page source
        page_source = driver.page_source

    # Parse with BeautifulSoup
    soup = BeautifulSoup(page_source, "lxml")

    # Find national team information
    national_team = None
    # Find <strong> tag containing "National Team:"
    strong_tag = soup.find("strong", string="National Team:")
    if strong_tag:
        # Get the parent <p> tag
        p_tag = strong_tag.find_parent("p")
        if p_tag:
            national_team_link = p_tag.find("a")
            if national_team_link:
                national_team = national_team_link.text.strip()

    # Find the "Match Logs (Summary)" paragraph
    match_logs_header = soup.find("p", class_="listhead", string="Match Logs (Summary)")

    if not match_logs_header:
        raise ValueError("Match Logs (Summary) section not found on the page")

    # Find the next ul element after the header
    match_logs_ul = match_logs_header.find_next("ul")

    if not match_logs_ul:
        raise ValueError("No match logs list found")

    # Extract all links from the ul, excluding national team competition pages
    season_links = []
    for li in match_logs_ul.find_all("li"):
        a_tag = li.find("a")
        if a_tag and a_tag.get("href"):
            href = a_tag.get("href")
            # Skip national team competition pages (nat_tm in URL)
            if "nat_tm" not in href.lower():
                full_url = f"https://fbref.com{href}"
                season_links.append(full_url)

    if not season_links:
        raise ValueError("No season match logs found")

    # Fetch match logs from each season
    all_dfs = []

    with get_driver() as driver:
        for season_url in season_links:
            try:
                driver.get(season_url)
                time.sleep(1)

                # Get page source
                season_page_source = driver.page_source

                # Parse with BeautifulSoup
                season_soup = BeautifulSoup(season_page_source, "lxml")

                # Find table with id 'matchlogs_all'
                table = season_soup.find("table", {"id": "matchlogs_all"})

                if table:
                    # Convert table to pandas DataFrame
                    season_df = pd.read_html(str(table))[0]

                    # Flatten multi-level column headers
                    if isinstance(season_df.columns, pd.MultiIndex):
                        # Combine multi-level headers into single level
                        season_df.columns = [
                            (
                                col[1]
                                if "Unnamed" in str(col[0])
                                else "_".join(col).strip("_")
                            )
                            for col in season_df.columns.values
                        ]

                    # Preprocessing: Remove invalid rows

                    # Find and remove rows where Pos column is "On matchday squad, but did not play"
                    if "Pos" in season_df.columns:
                        season_df = season_df[
                            season_df["Pos"] != "On matchday squad, but did not play"
                        ]

                    # Find Date column (could be named differently)
                    date_col = None
                    for col in season_df.columns:
                        if "Date" in col:
                            date_col = col
                            break

                    if date_col:
                        # Remove rows where Date is empty or contains "Date"
                        season_df = season_df[
                            season_df[date_col].notna()
                            & (season_df[date_col] != "Date")
                        ]

                    all_dfs.append(season_df)
            except Exception as e:
                # Log the error but continue with other seasons
                st.error(f"Error fetching {season_url}: {str(e)}")
                continue

    if not all_dfs:
        raise ValueError("No match logs data could be retrieved from any season")

    # Merge all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Remove national team rows if requested
    if remove_national_team and national_team:
        # Check if Squad or Opponent columns exist
        squad_col = None
        opponent_col = None

        # Find columns that contain 'Squad' or 'Opponent'
        for col in combined_df.columns:
            if "Squad" in col:
                squad_col = col
            if "Opponent" in col:
                opponent_col = col

        # Filter out rows where Squad or Opponent contains national team name
        if squad_col:
            combined_df = combined_df[
                ~combined_df[squad_col]
                .astype(str)
                .str.contains(national_team, na=False, case=False)
            ]
        if opponent_col:
            combined_df = combined_df[
                ~combined_df[opponent_col]
                .astype(str)
                .str.contains(national_team, na=False, case=False)
            ]

    # Remove duplicate rows
    combined_df = combined_df.drop_duplicates(ignore_index=True)

    return combined_df


# Input box for URL
url = st.text_input(
    "Enter FBref Player URL:",
    value="https://fbref.com/en/players/42fd9c7f/Kylian-Mbappe",
    help="Paste the FBref URL for any player",
)

if url:
    try:
        # Validate URL format
        if not url.startswith("https://fbref.com/en/players/"):
            st.error("Invalid URL format. Please use a valid FBref player URL.")
            st.stop()

        # # Convert to goallogs format if needed
        # if "/goallogs/" not in url:
        #     # Extract player ID and name from URL
        #     match = re.search(r"/players/([a-f0-9]+)/([^/]+)/?$", url)
        #     if match:
        #         player_id = match.group(1)
        #         player_name = match.group(2)
        #         url = f"https://fbref.com/en/players/{player_id}/goallogs/all_comps/{player_name}-Goal-Log"
        #         st.info(f"Converted URL to goal logs format")

        # Extract player name from URL for display
        player_name = "Player"
        if "/players/" in url:
            match = re.search(r"/players/([a-f0-9]+)/([^/]+)/?$", url)
            if match:
                player_id = match.group(1)
                player_name = match.group(2).replace("-", " ").title()
        
        # Fetch the page using cached function
        with st.spinner("Fetching match logs..."):
            # goal_logs_df = fetch_goal_logs(url)
            match_logs_df = fetch_match_logs(url, remove_national_team=True)

        st.success(f"Found {len(match_logs_df)} Matches!")
        
        # Show raw match logs before filtering
        raw_match_logs_df = match_logs_df.copy()
        
        # Add Season column to raw dataframe based on Date and 'Matchweek 1' in Round
        if 'Date' in raw_match_logs_df.columns and 'Round' in raw_match_logs_df.columns:
            raw_match_logs_df['Date'] = pd.to_datetime(raw_match_logs_df['Date'], errors='coerce')
            season_labels = []
            # Determine first valid year for initial season
            first_valid_year = None
            for dt in raw_match_logs_df['Date']:
                if pd.notna(dt):
                    first_valid_year = dt.year
                    break
            if first_valid_year is None:
                # Fallback: if no valid dates, leave Season empty strings
                raw_match_logs_df['Season'] = ''
            else:
                current_start_year = first_valid_year
                def format_season(start_year: int) -> str:
                    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"
                current_season = format_season(current_start_year)
                for i, row in raw_match_logs_df.iterrows():
                    round_val = str(row.get('Round', '')).strip()
                    date_val = row.get('Date')
                    # Start a new season at 'Matchweek 1'
                    if isinstance(round_val, str) and round_val.lower() == 'matchweek 1' and pd.notna(date_val):
                        current_start_year = int(date_val.year)
                        current_season = format_season(current_start_year)
                    season_labels.append(current_season)
                raw_match_logs_df['Season'] = pd.Series(season_labels, index=raw_match_logs_df.index).astype(str)
        else:
            # Ensure the column exists even if inputs are missing
            raw_match_logs_df['Season'] = ''

        # Propagate Season to working dataframe so it's selectable later
        if 'Season' in raw_match_logs_df.columns:
            try:
                match_logs_df['Season'] = raw_match_logs_df['Season']
            except Exception:
                # Fallback: align by index intersection
                common_idx = match_logs_df.index.intersection(raw_match_logs_df.index)
                match_logs_df.loc[common_idx, 'Season'] = raw_match_logs_df.loc[common_idx, 'Season']
        # show raw data
        # st.subheader("Raw Match Logs")
        # st.dataframe(raw_match_logs_df, use_container_width=True, hide_index=True)
        
        # Filter dataframe to keep only specified columns
        columns_to_keep = ['Date', 'Comp', 'Round', 'Squad', 'Opponent', 'Performance_Gls', 'Performance_PK', 'Expected_npxG', 'Season']
        match_logs_df = match_logs_df[columns_to_keep]
        top5_league_ucl = ['Champions Lg', 'Europa Lg',
                            'La Liga', 'Premier League', 'Serie A', 'Bundesliga', 'Ligue 1']
        match_logs_df = match_logs_df[match_logs_df['Comp'].isin(top5_league_ucl)]
        
        # Let user choose how many previous seasons to include (max 5)
        st.markdown("---")
        num_seasons = st.slider("Number of previous seasons to include", min_value=1, max_value=5, value=1)

        # Select only the last N seasons using the 'Season' column
        
        # Ensure Date is datetime
        # match_logs_df['Date'] = pd.to_datetime(match_logs_df['Date'], errors='coerce')
        if 'Season' in match_logs_df.columns and match_logs_df['Season'].astype(str).str.len().gt(0).any():
            # Identify the most recent seasons by max date per season
            season_order = (
                match_logs_df.dropna(subset=['Date'])
                              .groupby('Season')['Date']
                              .max()
                              .sort_values(ascending=False)
            )
            selected_seasons = list(season_order.index[:num_seasons])
            match_logs_df = match_logs_df[match_logs_df['Season'].isin(selected_seasons)]
        
        # Deduplicate rows that become identical after column selection
        # Prefer rows where Expected_npxG is present over missing
        key_cols = ['Date', 'Comp', 'Round', 'Squad', 'Opponent']
        tmp = match_logs_df.copy()
        tmp['_npxg_num'] = pd.to_numeric(tmp['Expected_npxG'], errors='coerce')
        tmp['_has_npxg'] = tmp['_npxg_num'].notna().astype(int)
        tmp = tmp.sort_values(key_cols + ['_has_npxg'], ascending=[True, True, True, True, True, False])
        tmp = tmp.drop_duplicates(subset=key_cols, keep='first')
        match_logs_df = tmp.drop(columns=['_npxg_num', '_has_npxg'])
        
        # Sort by date to ensure chronological order
        match_logs_df = match_logs_df.sort_values('Date').reset_index(drop=True)
        
        # st.markdown("---")

        # Display the last N seasons filtered dataframe
        # st.subheader(f"Last {num_seasons} Season{'s' if num_seasons > 1 else ''} Match Logs")
        # st.dataframe(match_logs_df, use_container_width=True, hide_index=True)
        
        # Color customization section
        st.markdown("---")
        st.subheader("🎨 Customize Chart Colors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            npgoals_color = st.color_picker(
                "Non-Penalty Goals Color", 
                value="#00fff2",
                help="Color for the npGoals line and overperforming areas"
            )
        
        with col2:
            npxg_color = st.color_picker(
                "None-Penalty xG Color", 
                value="#e217f7",
                help="Color for the npxG line and underperforming areas"
            )
        
        # Create chart dataframe with rolling averages
        st.markdown("---")
        st.subheader("📊 Rolling Average Performance Chart")
        
        # Calculate rolling averages
        chart_df = match_logs_df.copy()
        
        # Convert numeric columns to proper numeric types
        chart_df['Performance_Gls'] = pd.to_numeric(chart_df['Performance_Gls'], errors='coerce')
        chart_df['Performance_PK'] = pd.to_numeric(chart_df['Performance_PK'], errors='coerce')
        chart_df['Expected_npxG'] = pd.to_numeric(chart_df['Expected_npxG'], errors='coerce')
        
        # Calculate rolling averages
        chart_df['Rolling_avg_npGoals'] = (chart_df['Performance_Gls'] - chart_df['Performance_PK']).rolling(window=10, min_periods=1).mean()
        chart_df['Rolling_avg_Expected_npxG'] = chart_df['Expected_npxG'].rolling(window=10, min_periods=1).mean()
        chart_df['Match_Number'] = range(1, len(chart_df) + 1)
        
        # Create the line chart using matplotlib
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(19.8, 10.8))
        
        # Plot the lines
        ax.plot(chart_df['Match_Number'], chart_df['Rolling_avg_npGoals'], 
                color=npgoals_color, linewidth=2, label='Rolling Avg Non-Penalty Goals')
        ax.plot(chart_df['Match_Number'], chart_df['Rolling_avg_Expected_npxG'], 
                color=npxg_color, linewidth=2, label='Rolling Avg Expected npxG')
        
        # If multiple seasons selected, mark season boundaries and annotate season names
        if 'Season' in chart_df.columns and chart_df['Season'].nunique() > 1:
            # Identify start index of each season in the chart_df order
            season_start_mask = chart_df['Season'].ne(chart_df['Season'].shift())
            season_starts = chart_df.loc[season_start_mask, ['Match_Number', 'Season']]
            # Compute y position for annotations
            y_max = float(max(chart_df['Rolling_avg_npGoals'].max(), chart_df['Rolling_avg_Expected_npxG'].max()))
            y_min = float(min(chart_df['Rolling_avg_npGoals'].min(), chart_df['Rolling_avg_Expected_npxG'].min()))
            y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
            for idx, row in season_starts.iterrows():
                x_pos = row['Match_Number']
                season_label = str(row['Season'])
                # Skip the very first season's line but still annotate
                if x_pos != 1:
                    ax.axvline(x=x_pos, linestyle='--', color='white', alpha=0.75, linewidth=1)
                # Place season text slightly to the right of start
                ax.text(x_pos + 0.2, y_max + y_pad, season_label, color='white', fontsize=10, va='bottom')

        # Fill areas between lines
        # Custom color area where npGoals > Expected npxG (overperforming)
        ax.fill_between(chart_df['Match_Number'], 
                       chart_df['Rolling_avg_npGoals'], 
                       chart_df['Rolling_avg_Expected_npxG'],
                       where=(chart_df['Rolling_avg_npGoals'] >= chart_df['Rolling_avg_Expected_npxG']),
                       color=npgoals_color, alpha=0.3, interpolate=True)
        
        # Custom color area where Expected npxG > npGoals (underperforming)
        ax.fill_between(chart_df['Match_Number'], 
                       chart_df['Rolling_avg_npGoals'], 
                       chart_df['Rolling_avg_Expected_npxG'],
                       where=(chart_df['Rolling_avg_npGoals'] < chart_df['Rolling_avg_Expected_npxG']),
                       color=npxg_color, alpha=0.3, interpolate=True)
        
        # Customize the plot with dark theme
        # Set black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Remove grid and legend
        ax.grid(False)
        
        # Set y-axis text to white and hide x-axis completely
        ax.tick_params(axis='y', colors='white')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('10 match rolling average', fontsize=15, fontweight='bold', color='white', labelpad=15)
        
        # Create custom title with colored text using highlight_text
        
        title_text = f'{player_name} | <npGoals> vs <npxG>'
        
        fig_text(x=0.05, y=1, s=title_text, 
                highlight_textprops=[
                    {"color": npgoals_color, "fontweight": "bold"},
                    {"color": npxg_color, "fontweight": "bold"}
                ],
                fontsize=30, fontweight='bold', ha='left', va='center',
                color='white', transform=fig.transFigure)

        fig.text(0.05, 0.95, "Data: fbref  |  Made by: Mohammad Adnan (@adnaaan433) For the Real Deal Podcast (@Realdealpodz)", color='white', 
                    fontsize=15, ha='left', va='center')
        
        # # Add logos to top right corner
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from PIL import Image
        
        # Load and add RealDeal logo
        try:
            realdeal_logo = plt.imread("RealDeal_Logo.png")
            # st.text(f"RealDeal logo loaded successfully, shape: {realdeal_logo.shape}")
            imagebox1 = OffsetImage(realdeal_logo, zoom=0.12)  # Increased zoom for visibility
            ab1 = AnnotationBbox(imagebox1, (0.875, 1.05), frameon=False, 
                               xycoords='figure fraction', box_alignment=(0, 1))
            ax.add_artist(ab1)
            # st.text("RealDeal logo added to chart")
        except Exception as e:
            st.text(f"Could not load RealDeal logo: {e}")
        
        # # Load and add Twitter handle logo
        # try:
        #     twitter_logo = plt.imread(r"adnaaan433_twitter_handle.png")
        #     # st.text(f"Twitter logo loaded successfully, shape: {twitter_logo.shape}")
        #     imagebox2 = OffsetImage(twitter_logo, zoom=0.12)  # Increased zoom for visibility
        #     ab2 = AnnotationBbox(imagebox2, (0.9, 0), frameon=False, 
        #                        xycoords='figure fraction', box_alignment=(0, 1))
        #     ax.add_artist(ab2)
        #     # st.text("Twitter logo added to chart")
        # except Exception as e:
        #     st.text(f"Could not load Twitter logo: {e}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)

        # Download PNG option
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='black', dpi=100)
        buf.seek(0)
        
        # Create download button for PNG
        st.download_button(
            label="📥 Download Chart as PNG",
            data=buf.getvalue(),
            file_name=f"{player_name.replace(' ', '_')}_performance_chart.png",
            mime="image/png",
        )

    except ValueError as e:
        st.warning(str(e))
        st.info("The player may not have any logs recorded.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback

        with st.expander("Error details"):
            st.code(traceback.format_exc())
else:
    st.info("👆 Enter a URL above to get started")
