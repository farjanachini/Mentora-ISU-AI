import unittest
from unittest.mock import patch

import website_retrieval


class WebsiteRetrievalTests(unittest.TestCase):
    def test_parser_extracts_visible_text_and_internal_links(self):
        parser = website_retrieval._PageParser()
        parser.feed(
            "<html><script>hidden text</script><body>"
            "<h1>Faculty Members</h1>"
            "<a href='/faculty/123'>Dr. Hakikur Rahman</a>"
            "</body></html>"
        )

        text = "".join(parser.text)
        self.assertIn("Faculty Members", text)
        self.assertNotIn("hidden text", text)
        self.assertEqual(
            parser.links,
            [("/faculty/123", "Dr. Hakikur Rahman")]
        )

    def test_parser_attaches_official_faculty_section_to_each_card(self):
        parser = website_retrieval._PageParser()
        parser.feed(
            '<h4 class="details-page-heading">Adjunct Faculties</h4>'
            '<div class="col-md-10 card">'
            '<h4 class="name">Md. Wahiduzzaman Wahid</h4>'
            "<h5>Chief Technology Officer, Standard Group</h5>"
            "</div>"
        )

        text = "".join(parser.text)
        self.assertIn("Faculty category: Adjunct Faculties", text)
        self.assertIn("Name: Md. Wahiduzzaman Wahid", text)

    def test_matching_faculty_category_is_associated_with_queried_person(self):
        context = "\n".join([
            "Source URL: https://isu.ac.bd/faculties/department-of-computer-science-and-engineering",
            "Information:",
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            "Faculty category: Adjunct Faculties",
            "Name: Md. Wahiduzzaman Wahid",
            "Personal title: Chief Technology Officer, Standard Group",
        ])

        matches = website_retrieval.matching_faculty_categories(
            "Tell me about Wahiduzzaman Wahid",
            context
        )

        self.assertEqual(matches, [{
            "name": "Md. Wahiduzzaman Wahid",
            "category": "Adjunct Faculties",
            "department": "Department of Computer Science & Engineering",
        }])

    def test_relevant_excerpt_preserves_category_heading(self):
        text = "\n".join([
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            "Adjunct Faculties",
            "Faculty category: Adjunct Faculties",
            "Mohshin Uddin Anwar",
            "Managing Director, ZASS Engineering & Development Ltd.",
            "BSEngg. (CSE, BUET), MSEngg. (ICT, BUET), MBA (IBA)",
            "Prof. Mohammad Shafiul Alam, PhD",
            "Adjunct Professor, Department of Computer Science and Engineering",
            "PhD (BUET), Dhaka, Bangladesh",
            "Md. Wahiduzzaman Wahid",
            "Chief Technology Officer, Standard Group",
        ])

        excerpt = website_retrieval._relevant_excerpt(
            text,
            website_retrieval._terms("Wahiduzzaman Wahid")
        )

        self.assertIn(
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            excerpt
        )
        self.assertIn("Adjunct Faculties", excerpt)
        self.assertIn("Faculty category: Adjunct Faculties", excerpt)
        self.assertIn("Chief Technology Officer, Standard Group", excerpt)

    def test_url_filter_accepts_all_isu_subdomains_only(self):
        for url in (
            "https://www.isu.ac.bd/tuition-fees",
            "https://isu.ac.bd/tuition-fees",
            "https://convocation.isu.ac.bd/",
            "https://admission.isu.ac.bd/tuition-fees",
            "https://portal.student.isu.ac.bd/",
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    website_retrieval._normalize_url(url),
                    url
                )

        for url in (
            "https://isu.ac.bd.example.com/faculty",
            "https://notisu.ac.bd/faculty",
            "https://schedule.convocation.isu.ac.bd.example.com/",
        ):
            with self.subTest(url=url):
                self.assertIsNone(website_retrieval._normalize_url(url))

    def test_all_department_specific_pages_have_equal_priority(self):
        query_terms = website_retrieval._terms(
            "Tell me about Dr. Md Hakikur Rahman"
        )
        cse_page = {
            "url": (
                "https://www.isu.ac.bd/faculties/"
                "department-of-computer-science-and-engineering"
            ),
            "text": "Dr. Md Hakikur Rahman Professor and Head",
        }
        directory_page = {
            "url": "https://www.isu.ac.bd/faculties/all-departments",
            "text": "Dr. Md Hakikur Rahman Professor and Head",
        }
        ai_page = {
            "url": (
                "https://www.isu.ac.bd/faculties/"
                "department-of-ai-and-data-science"
            ),
            "text": "Dr. Md Hakikur Rahman Professor and Head",
        }

        self.assertEqual(
            website_retrieval._page_score(cse_page, query_terms),
            website_retrieval._page_score(ai_page, query_terms)
        )
        self.assertGreater(
            website_retrieval._page_score(cse_page, query_terms),
            website_retrieval._page_score(directory_page, query_terms)
        )

    def test_retrieval_follows_live_links_to_matching_profile(self):
        home_page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University",
            "links": [
                ("/faculties/all-departments", "Faculty Members")
            ],
        }
        faculty_page = {
            "url": "https://isu.ac.bd/faculties/all-departments",
            "text": (
                "Dr. Md. Hakikur Rahman Professor and Head "
                "Department of AI and Data Science "
                "DSc Manipur International University"
            ),
            "links": [
                (
                    "/faculty/4635/Dr-Md-Hakikur-Rahman",
                    "Dr. Md. Hakikur Rahman"
                )
            ],
        }
        profile_page = {
            "url": "https://isu.ac.bd/faculty/4635/Dr-Md-Hakikur-Rahman",
            "text": (
                "Dr. Md. Hakikur Rahman Professor and Head "
                "Department of AI and Data Science "
                "DSc Manipur International University"
            ),
            "links": [],
        }

        def fake_fetch_batch(urls):
            if tuple(urls) == website_retrieval.START_URLS:
                return [home_page]
            if any("all-departments" in url for url in urls):
                return [faculty_page]
            if any("Hakikur" in url for url in urls):
                return [profile_page]
            return []

        with patch.object(
            website_retrieval, "_fetch_batch", side_effect=fake_fetch_batch
        ):
            context = website_retrieval.retrieve_documents(
                "Tell me about Dr. Md Hakikur Rahman"
            )

        self.assertIn("Source URL: https://isu.ac.bd/faculty/4635/", context)
        self.assertIn("Manipur International University", context)

    def test_retrieval_includes_convocation_subdomain_date(self):
        main_page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University",
            "links": [],
        }
        cse_page = {
            "url": (
                "https://isu.ac.bd/faculties/"
                "department-of-computer-science-and-engineering"
            ),
            "text": "International Standard University Convocation",
            "links": [],
        }
        ai_page = {
            "url": (
                "https://isu.ac.bd/faculties/"
                "department-of-ai-and-data-science"
            ),
            "text": "International Standard University Convocation",
            "links": [],
        }
        convocation_page = {
            "url": "https://convocation.isu.ac.bd/",
            "text": (
                "International Standard University 1st Convocation "
                "Ceremony Date 03 April 2027"
            ),
            "links": [],
        }

        with patch.object(
            website_retrieval,
            "_fetch_batch",
            return_value=[main_page, cse_page, ai_page, convocation_page]
        ) as fetch_batch:
            context = website_retrieval.retrieve_documents(
                "What is the exact convocation date?"
            )

        fetch_batch.assert_called_once_with(website_retrieval.START_URLS)
        self.assertIn(
            "Source URL: https://convocation.isu.ac.bd/",
            context
        )
        self.assertIn("03 April 2027", context)

    def test_retrieval_discovers_and_follows_linked_isu_subdomains(self):
        main_page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University",
            "links": [
                (
                    "https://portal.student.isu.ac.bd/",
                    "Portal",
                )
            ],
        }
        cse_page = {
            "url": (
                "https://isu.ac.bd/faculties/"
                "department-of-computer-science-and-engineering"
            ),
            "text": "International Standard University Quality Assurance",
            "links": [],
        }
        ai_page = {
            "url": (
                "https://isu.ac.bd/faculties/"
                "department-of-ai-and-data-science"
            ),
            "text": "International Standard University Quality Assurance",
            "links": [],
        }
        portal_page = {
            "url": "https://portal.student.isu.ac.bd/",
            "text": "Quality assurance policy documents are available here.",
            "links": [],
        }

        def fake_fetch_batch(urls):
            if tuple(urls) == website_retrieval.START_URLS:
                return [main_page, cse_page, ai_page]
            if any("portal.student.isu.ac.bd" in url for url in urls):
                return [portal_page]
            return []

        with patch.object(
            website_retrieval, "_fetch_batch", side_effect=fake_fetch_batch
        ):
            context = website_retrieval.retrieve_documents(
                "Where can I find the quality assurance policy?"
            )

        self.assertIn(
            "Source URL: https://portal.student.isu.ac.bd/",
            context
        )
        self.assertIn("Quality assurance policy", context)

    def test_subdomain_roots_are_crawled_before_deep_pages(self):
        main_page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University",
            "links": [
                (
                    "https://portal.isu.ac.bd/quality-assurance-policy",
                    "Quality assurance policy",
                ),
                ("https://info.isu.ac.bd/", "Information portal"),
            ],
        }
        portal_page = {
            "url": "https://info.isu.ac.bd/",
            "text": "Information portal",
            "links": [],
        }

        def fake_fetch_batch(urls):
            if tuple(urls) == website_retrieval.START_URLS:
                return [main_page]
            return [portal_page] if "https://info.isu.ac.bd/" in urls else []

        with patch.object(
            website_retrieval, "MAX_PAGES", 2
        ), patch.object(
            website_retrieval, "_fetch_batch", side_effect=fake_fetch_batch
        ) as fetch_batch:
            website_retrieval.retrieve_documents(
                "Where is the quality assurance policy?"
            )

        self.assertIn(
            "https://info.isu.ac.bd/",
            fetch_batch.call_args_list[1].args[0]
        )

    def test_multiple_department_entries_for_person_are_preserved(self):
        root_page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University",
            "links": [
                (
                    "/faculties/department-of-ai-and-data-science",
                    "Department of AI and Data Science"
                )
            ],
        }
        cse_page = {
            "url": (
                "https://isu.ac.bd/faculties/"
                "department-of-computer-science-and-engineering"
            ),
            "text": (
                "Dr. Md. Hakikur Rahman\n"
                "Professor and Head, Department of Computer Science & Engineering\n"
                "DSc. in AI and Data Science (ongoing)"
            ),
            "links": [],
        }
        directory_page = {
            "url": "https://isu.ac.bd/faculties/all-departments",
            "text": "Faculty Members",
            "links": [
                (
                    "/faculties/department-of-ai-and-data-science",
                    "Department of AI and Data Science"
                )
            ],
        }
        ai_page = {
            "url": "https://isu.ac.bd/faculties/department-of-ai-and-data-science",
            "text": (
                "Dr. Md. Hakikur Rahman\n"
                "Professor and Head, Department of AI & Data Science"
            ),
            "links": [],
        }

        def fake_fetch_batch(urls):
            if tuple(urls) == website_retrieval.START_URLS:
                return [root_page, cse_page, ai_page, directory_page]
            if any("department-of-ai-and-data-science" in url for url in urls):
                return [ai_page]
            return []

        with patch.object(
            website_retrieval, "_fetch_batch", side_effect=fake_fetch_batch
        ):
            context = website_retrieval.retrieve_documents(
                "Tell me about Dr. Md Hakikur Rahman"
            )

        self.assertIn(
            "Professor and Head, Department of Computer Science & Engineering",
            context
        )
        self.assertIn(
            "Professor and Head, Department of AI & Data Science",
            context
        )

    def test_retrieval_fails_instead_of_using_offline_information(self):
        with patch.object(website_retrieval, "_fetch_batch", return_value=[]):
            with self.assertRaises(website_retrieval.WebsiteRetrievalError):
                website_retrieval.retrieve_documents("ISU admission info")

    def test_retrieval_returns_no_sources_when_no_page_matches(self):
        page = {
            "url": "https://isu.ac.bd/",
            "text": "International Standard University campus and faculty",
            "links": [],
        }

        with patch.object(
            website_retrieval, "_fetch_batch", return_value=[page]
        ):
            context = website_retrieval.retrieve_documents(
                "What is photosynthesis?"
            )

        self.assertEqual(context, "")


if __name__ == "__main__":
    unittest.main()
