import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class ChatApiTests(unittest.TestCase):
    def test_all_faculty_question_uses_room_for_complete_list(self):
        captured_config = {}

        def generate_content(**kwargs):
            captured_config["config"] = kwargs["config"]
            return SimpleNamespace(
                text="Prof. A — Professor.\nDr. B — Lecturer."
            )

        with patch.object(
            app, "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            ))
        ), patch.object(
            app, "retrieve_documents",
            return_value="Source URL: https://isu.ac.bd/faculties/cse"
        ):
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "List all CSE faculty members"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Dr. B", response.get_json()["response"])
        self.assertGreaterEqual(
            captured_config["config"].max_output_tokens,
            1000
        )

    def test_greeting_skips_website_retrieval_and_has_no_sources(self):
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="Hello! How can I help?"
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(app, "retrieve_documents") as retrieve_documents:
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "Hi!"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["sources"], [])
        retrieve_documents.assert_not_called()

    def test_chat_adds_matched_adjunct_category_if_model_omits_it(self):
        context = "\n".join([
            "Source URL: https://isu.ac.bd/faculties/department-of-computer-science-and-engineering",
            "Information:",
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            "Faculty category: Adjunct Faculties",
            "Name: Md. Wahiduzzaman Wahid",
            "Personal title: Chief Technology Officer, Standard Group",
        ])
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text=(
                    "Role: Chief Technology Officer, Standard Group. "
                    "Qualifications: BUET, DU, OCP."
                )
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(
            app, "retrieve_documents", return_value=context
        ):
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "Tell me about Wahiduzzaman Wahid"}
            )

        self.assertEqual(response.status_code, 200)
        answer = response.get_json()["response"]
        self.assertIn("an adjunct faculty member", answer)
        self.assertIn("Department of Computer Science & Engineering", answer)


if __name__ == "__main__":
    unittest.main()
