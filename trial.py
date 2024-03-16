from bs4 import BeautifulSoup

# Sample HTML code
html = """
<div class="col-lg-4 col-md-4 col-sm-6 my-2 reportsDownload" data-cat="capital-market" data-section="equities" data-type="archives">
	<div class="card h-100">
		<div class="card-body">
			<label class="chk_container">
				CM - Security-wise Delivery Positions
				<input type="checkbox">
				<span class="checkmark"></span>
			</label>
			<span class="reportDownloadIcon">
				<a aria-label="Download File" onclick="SingledownloadReports('#cr_equity_archives', 'equities', this)" href="javascript:;" class="pdf-download-link"></a>
			</span>
		</div>
	</div>
</div>
"""

# Parse the HTML
soup = BeautifulSoup(html, "html.parser")

# Find the <a> element within the specified class
download_link_element = soup.find("a", class_="pdf-download-link")

# Extract the href attribute (download link)
if download_link_element:
    download_link = download_link_element.get("href")
    print("Download Link:", download_link)
else:
    print("Download link not found.")
