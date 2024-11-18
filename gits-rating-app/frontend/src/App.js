// frontend/src/App.js

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import { styled } from '@mui/system';
import axios from 'axios';

// Styled Components
const UploadBox = styled(Box)(({ theme }) => ({
  border: '2px dashed #ccc',
  borderRadius: '10px',
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  marginBottom: theme.spacing(2),
}));

const Input = styled('input')({
  display: 'none',
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [rating, setRating] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setError(null);
    setRating(null);
    const file = event.target.files[0];
    if (file && file.type === 'image/gif') {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    } else {
      setError('Please select a valid GIF file.');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('No file selected.');
      return;
    }

    const formData = new FormData();
    formData.append('gif', selectedFile);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/rate-gif', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setRating(response.data.rating);
    } catch (err) {
      console.error(err);
      setError('Error uploading file.');
    } finally {
      setLoading(false);
    }
  };

  return (
      <Container maxWidth="sm" style={{ marginTop: '50px' }}>
        <Typography variant="h4" gutterBottom align="center">
          GIF Rating App
        </Typography>

        {error && (
            <Alert severity="error" style={{ marginBottom: '20px' }}>
              {error}
            </Alert>
        )}

        <UploadBox>
          <label htmlFor="contained-button-file">
            <Input
                accept="image/gif"
                id="contained-button-file"
                type="file"
                onChange={handleFileChange}
            />
            <Button variant="contained" component="span">
              Choose GIF
            </Button>
          </label>
        </UploadBox>

        {preview && (
            <Box textAlign="center" marginBottom="20px">
              <img
                  src={preview}
                  alt="Selected GIF"
                  style={{ maxWidth: '100%', maxHeight: '300px' }}
              />
            </Box>
        )}

        <Box textAlign="center" marginBottom="20px">
          <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={loading || !selectedFile}
          >
            {loading ? <CircularProgress size={24} /> : 'Rate GIF'}
          </Button>
        </Box>

        {rating && (
            <Alert severity="success" style={{ textAlign: 'center' }}>
              The GIF rating is: {rating}
            </Alert>
        )}
      </Container>
  );
}

export default App;
