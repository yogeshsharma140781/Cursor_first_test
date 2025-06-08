import React, { useState } from "react";
import {
  Container,
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
  Grid,
  Paper,
  CssBaseline,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";

const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "ar", label: "Arabic" },
  { value: "zh-cn", label: "Chinese (Simplified)" },
  { value: "zh-tw", label: "Chinese (Traditional)" },
  { value: "nl", label: "Dutch" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "it", label: "Italian" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "pl", label: "Polish" },
  { value: "pt", label: "Portuguese" },
  { value: "ru", label: "Russian" },
  { value: "es", label: "Spanish" },
  { value: "tr", label: "Turkish" },
  { value: "uk", label: "Ukrainian" },
  { value: "vi", label: "Vietnamese" },
];

const theme = createTheme({
  typography: {
    fontFamily: "Inter, Arial, sans-serif",
    h4: {
      fontWeight: 700,
      color: "#1e90ff",
      letterSpacing: 0.5,
    },
  },
  palette: {
    primary: { main: "#1e90ff" },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
          fontSize: 16,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          background: "#fafbfc",
          borderRadius: 8,
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          background: "#fff",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
      },
    },
  },
});

export default function App() {
  const [tab, setTab] = useState(0);
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("en");
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");

  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;
  const charCount = inputText.length;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: "100vh", bgcolor: "#fff" }}>
        <Container maxWidth="sm" sx={{ py: 5 }}>
          {/* Header */}
          <Box display="flex" alignItems="center" justifyContent="center" mb={4}>
            <Typography variant="h4">Translator AI</Typography>
          </Box>

          {/* Tabs */}
          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            variant="fullWidth"
            sx={{
              mb: 3,
              borderRadius: "24px",
              background: "#f5f6fa",
              minHeight: 44,
              "& .MuiTabs-indicator": {
                background: "#1e90ff",
                height: 3,
                borderRadius: 2,
              },
            }}
          >
            <Tab
              label="Text"
              sx={{
                fontWeight: 500,
                fontSize: 16,
                color: tab === 0 ? "#1e90ff" : "#888",
                minHeight: 44,
              }}
            />
            <Tab
              label="Document (pdf, docx)"
              disabled
              sx={{
                fontWeight: 500,
                fontSize: 16,
                color: "#bbb",
                minHeight: 44,
              }}
            />
          </Tabs>

          {tab === 0 && (
            <Paper elevation={0} sx={{ p: 3, bgcolor: "#fff" }}>
              <Grid container spacing={2} direction="column">
                <Grid item>
                  <FormControl fullWidth>
                    <InputLabel id="source-lang-label">Source Language</InputLabel>
                    <Select
                      labelId="source-lang-label"
                      value={sourceLang}
                      label="Source Language"
                      onChange={e => setSourceLang(e.target.value)}
                    >
                      <MenuItem value="auto">Detect Language</MenuItem>
                      {LANGUAGES.map(lang => (
                        <MenuItem key={lang.value} value={lang.value}>
                          {lang.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item>
                  <TextField
                    multiline
                    minRows={6}
                    maxRows={10}
                    fullWidth
                    placeholder="Type or paste text here..."
                    value={inputText}
                    onChange={e => setInputText(e.target.value)}
                    variant="outlined"
                  />
                </Grid>
                <Grid item>
                  <Typography variant="body2" color="#888" sx={{ mt: -1, mb: 0, fontSize: 13 }}>
                    {wordCount} words, {charCount} characters
                  </Typography>
                </Grid>
                <Grid item>
                  <Grid container spacing={2} alignItems="flex-end">
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel id="target-lang-label">Target Language</InputLabel>
                        <Select
                          labelId="target-lang-label"
                          value={targetLang}
                          label="Target Language"
                          onChange={e => setTargetLang(e.target.value)}
                        >
                          {LANGUAGES.map(lang => (
                            <MenuItem key={lang.value} value={lang.value}>
                              {lang.label}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="contained"
                        sx={{
                          bgcolor: "#1e90ff",
                          color: "#fff",
                          minHeight: 44,
                          "&:hover": { bgcolor: "#005bb5" },
                        }}
                        onClick={() =>
                          setOutputText(inputText ? `Translated: ${inputText}` : "")
                        }
                      >
                        Translate
                      </Button>
                    </Grid>
                  </Grid>
                </Grid>
                <Grid item>
                  <TextField
                    multiline
                    minRows={6}
                    maxRows={10}
                    fullWidth
                    placeholder="Translation will appear here..."
                    value={outputText}
                    InputProps={{
                      readOnly: true,
                    }}
                    variant="outlined"
                  />
                </Grid>
              </Grid>
            </Paper>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}