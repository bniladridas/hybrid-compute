# Hybrid Compute API Design Guide

## Version 1.0.0 - Published 2026 Feb 19

_Internal version published 2026 Feb 1_

- [Documentation Index](docs/index.md)
- [Compatibility Guide](docs/COMPATIBILITY.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [API Test Results](api/TEST_RESULTS.md)

## Contents
- [Purpose](#purpose)
- [Overview](#overview)
    - [Why Have an API Design Guide?](#why-have-an-api-design-guide)
    - [Guiding Principles](#guiding-principles)
- [API Design Essentials](#api-design-essentials)
    - [Requests](#requests)
        - [URIs/Paths](#urispaths)
            - [Use Plural Resource Names](#use-plural-resource-names)
            - [Use Nouns, Not Verbs in the Base URI](#use-nouns-not-verbs-in-the-base-uri)
            - [Shorten Associations in the URI - Hide Dependencies in the Parameter List](#shorten-associations-in-the-uri-hide-dependencies-in-the-parameter-list)
        - [Uniform Interface with HTTP Verbs/Methods](#uniform-interface-with-http-verbsmethods)
    - [Resource Identifiers](#resource-identifiers)
        - [Do Not Use Database Table Row IDs as Resource IDs](#do-not-use-database-table-row-ids-as-resource-ids)
        - [SQUUID Implementations](#squuid-implementations)
    - [Versioning](#versioning)
        - [Why Versioning?](#why-versioning)
        - [Versioning Methods](#versioning-methods)
            - [Version in the HTTP `Accept` Header](#version-in-the-http-accept-header)
            - [Version in the URI](#version-in-the-uri)
            - [Versioning Schemes](#versioning-schemes)
            - [Breaking Changes](#breaking-changes)
            - [Non-Breaking Changes](#non-breaking-changes)
        - [API Versioning Summary](#api-versioning-summary)
    - [Pagination](#pagination)
        - [Why Pagination?](#why-pagination)
        - [Pagination - Summary](#pagination-summary)
        - [Pagination Method](#pagination-method)
        - [Pagination - Leading the Consumer](#pagination-leading-the-consumer)
        - [Client Use of Pagination URIs](#client-use-of-pagination-uris)
    - [Filter/Sort/Search](#filtersortsearch)
    - [Content Negotiation](#content-negotiation)
    - [Security](#security)
        - [Transport](#transport)
        - [Authentication/Authorization](#authenticationauthorization)
        - [CORS (Cross-Origin Resource Sharing)](#cors-cross-origin-resource-sharing)
    - [Responses](#responses)
        - [HTTP Status Codes](#http-status-codes)
        - [Error Handling / Messages](#error-handling-messages)
        - [Response Envelopes and Hypermedia](#response-envelopes-and-hypermedia)
            - [HAL Specification](#hal-specification)
        - [Response Meta Data](#response-meta-data)
    - [JSON](#json)
        - [Data Formatting](#data-formatting)
            - [Dates and Times](#dates-and-times)
            - [UTF-8](#utf-8)
            - [Currency and Fixed-Point Values](#currency-and-fixed-point-values)
    - [Performance](#performance)
        - [Consumer-Side](#consumer-side)
        - [Producer-Side](#producer-side)
            - [Condense JSON Response with HTTP Compression](#condense-json-response-with-http-compression)
            - [Conditional `GET` with Caching and `ETag`](#conditional-get-with-caching-and-etag)
    - [Testing](#testing)
        - [Improve API Test Performance](#improve-api-test-performance)
    - [REST API Reference](#rest-api-reference)
        - [Endpoints](#endpoints)
            - [Images](#images)
            - [Tiles](#tiles)
            - [Jobs](#jobs)
        - [Example Requests](#example-requests)
    - [Project Overview](#project-overview)
        - [Architecture](#architecture)
        - [Prerequisites](#prerequisites)
        - [Setup](#setup)
        - [Usage](#usage)
    - [Git Commit Standards](#git-commit-standards)
    - [License](#license)

## Purpose

The purpose of the Hybrid Compute API Design Guide is to provide standards and best practices for any REST API components within this project. This guide helps developers and architects design and implement consistent and well-documented APIs.

A basic knowledge of REST, HTTP, and JSON is assumed. We will provide links to resources for those who want to learn more about the fundamentals of these technologies.

## Overview

API Design has come of age, and has become a first-class citizen in the enterprise. The principles and practices that we follow determine the usability and overall quality of our APIs.

### Why Have an API Design Guide?

An API Design Guide provides us with the following:

- APIs that have a consistent look-and-feel.
- APIs that act in a predictable/expected manner. Don't surprise the consumer.
- Take the guesswork out of designing and implementing an API by following the design practices.
- Streamlined tooling and documentation.

### Guiding Principles

Here are the guiding principles for designing an API and determining the merit of each design practice:

- Use commonly accepted web standards (e.g., HTTP, JSON) when it makes sense to do so.
- Each API should be consistent with commonly accepted practices leveraged by other good APIs on the Internet.
- It should be simple to consume and test.
- It must be pragmatic and performant.

## API Design Essentials

The remaining sections of this page cover the key areas to consider when designing and implementing an API.

### Requests

#### URIs/Paths

The URI (Uniform Resource Indicator) / Path is the path to a resource exposed by an API. Here are the key principles:

- URIs should be intuitive. This property, known as affordance:
  - Makes an API easy to use and understand.
  - Reduces the amount of documentation needed for an API.
- Keep URIs simple and short.
- Shorten associations/dependencies in the URIs.
- Use plural resource names.
- Use Nouns, not Verbs in the Base URI.
  - Use HTTP Verbs - GET, POST, PUT, DELETE, and PATCH - to indicate the operation on a resource.

##### Use Plural Resource Names

Resource names should be plural, for example if we're exposing Images, then the Base URI should look like:

```
/images
```

unless the resource is a singleton, for example, the overall status of the system might be `/status`.

##### Use Nouns, Not Verbs in the Base URI

Never put a Verb in the Base URI. Rather than something like `/get_images_by_id`, we use the following URI with an HTTP GET:

```
/images/4
```

##### Shorten Associations in the URI - Hide Dependencies in the Parameter List

Resources are related to one another. Instead of:

```
users/1/orders/54/images/5900
```

Use:

```
users/1/images?order_id=54
```

#### Uniform Interface with HTTP Verbs/Methods

The following table shows the standard HTTP Verbs/Methods to act on resources:

| HTTP Verb / Method | Action             |
|:-------------------|:-------------------|
| `GET`              | Read               |
| `POST`             | Create             |
| `PUT`              | Update (full)      |
| `PATCH`            | Update (partial)  |
| `DELETE`           | Delete             |

### Resource Identifiers

Resources should use Sequential UUIDs (SQUUIDs).

SQUUIDs are stored as binary(16). SQUUIDs should be represented in string form as lowercase.

#### Do Not Use Database Table Row IDs as Resource IDs

Auto-incrementing integer database row identifiers should not be used as Resource IDs or exposed in any way through the API.

#### SQUUID Implementations

```
def squuid
  ary = SecureRandom.random_bytes(16).unpack("NnnnnN")
  ary[0] = Time.now.to_i
  ary[2] = (ary[2] & 0x0fff) | 0x4000
  ary[3] = (ary[3] & 0x3fff) | 0x8000
  "%08x-%04x-%04x-%04x-%04x%08x" % ary
end
```

### Versioning

#### Why Versioning?

API Versioning is an important aspect of API design because it informs the consumer about an API's capabilities and data. Consumers use the version number for compatibility.

#### Versioning Methods

Here are 2 supported methods of API versioning:

- **Version in the HTTP Accept Header.** (this is our preference)
- Version in the URI.

##### Version in the HTTP `Accept` Header

```
GET /images/4
Accept: application/vnd.hybrid-compute.v1+json
```

Pros:
- The version is a representation of a resource, and this information goes in the HTTP Accept Header.
- It leverages a mechanism already provided by the HTTP specification.
- It supports content-based load balancing.

##### Version in the URI

```
GET /v1/images/4
```

Pros:
- It works well and is widely used.
- It's simple and easy to read.
- It supports testing from a web browser.

Cons:
- A URI identifies the location of resource, and the URI shouldn't change just because the data changes.

#### Versioning Schemes

- We will use a Major and Minor version in the form `x.y`, where `x` >= 1 and represents the Major version number, and `y` >= 0 and represents the Minor version number.
- We never have a Major version of 0 because it makes the API appear unstable.
- There is no need to modify either the Major or Minor version for Non-Breaking changes.
- We change the Major version when there are multiple Breaking Changes.
- We change the Minor version when there are 1 or more Breaking Changes.

##### Breaking Changes

A Breaking Change is any change to an API that could break a contract:

- Deprecating a feature
- Refactoring a non-trivial portion of the API implementation
- When a field's data type changes
- Changing the Security model

##### Non-Breaking Changes

A Non-Breaking Change is a change that does not break a contract:

- Adding a new feature
- Upgrading documentation

#### API Versioning Summary

- Every API must have a version.
- Every API invocation must specify a version number.
- All new APIs will put the version in the HTTP Accept Header.
- Version format: `x.y` (Major.Minor)

### Pagination

#### Why Pagination?

An API must be able to control/gate the amount data returned in the response so that the Consumer is able to handle the volume of data.

#### Pagination - Summary

- Use offset and limit: `/images?offset=543&limit=25`
- Lead the Consumer through the API with links/URIs in the JSON response

#### Pagination Method

The `offset` is semantic - it could be a UUID, ID, a Date, etc. that is sortable.

#### Pagination - Leading the Consumer

```
GET /images?offset=100&limit=25

{
   "_links": {
     "self": { "href": "/images?offset=543&limit=25" },
     "next": { "href": "/images?offset=768&limit=25" },
     "prev": { "href": "/images?offset=123&limit=25" }
   },
   "items": []
}
```

This is an implementation of [HAL (JSON Hypermedia API Language)](https://tools.ietf.org/html/draft-kelly-json-hal-06).

#### Client Use of Pagination URIs

Clients should avoid trying to construct new URLs to the API, but rather depend on the `next` and `previous` links returned by the API.

### Filter/Sort/Search

- Keep it simple - just use HTTP parameters
- Use a separate field for each filter/sort/search parameter
- For sort, use an explicit `sort` parameter:

```
https://api.example.com/images.json?sort=created_at
```

### Content Negotiation

Content Negotiation is a mechanism that enables an API to serve a document/response in different formats. Based on current industry best practices, we prefer JSON.

```
Accept: application/json
```

The HTTP `Content-Type` Header:

```
Content-Type: application/json
```

### Security

#### Transport

Always use HTTPS to communicate with and between APIs.

#### Authentication/Authorization

- Authentication validates a service consumer (User, Mobile app, or Service)
- Authorization ensures a subject has permission to access services and resources

#### CORS (Cross-Origin Resource Sharing)

CORS defines a mechanism in which an API and its Consumers can collaborate to determine if it's safe to allow the cross-origin request.

| HTTP Response Header | Description |
| -------------------- | ----------- |
| `Access-Control-Allow-Origin` | Indicates whether a resource can be shared |
| `Access-Control-Allow-Credentials` | Indicates whether the response can include credentials |
| `Access-Control-Allow-Methods` | Indicates which HTTP Methods are allowed |

### Responses

When a Consumer invokes an API, one of three things can happen:
- Success - everything worked properly
- Consumer Error - The Consumer used bad data
- API Error - The API had a processing error

#### HTTP Status Codes

| HTTP Status Code | Meaning |
|:-----------------|:--------|
| 200 | OK |
| 201 | Created |
| 202 | Accepted |
| 204 | No Content |
| 301 | Moved Permanently |
| 304 | Not Modified |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 410 | Gone |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

#### Error Handling / Messages

For consumer-facing APIs:

```json
{
  "errors": [
    {
      "code": "unrecoverable_error",
      "title": "The flux capacitor disintegrated",
      "details": "Hold on, the end is nigh.",
      "user_message": "OMG, panic!"
    }
  ]
}
```

#### Response Envelopes and Hypermedia

We prefer [HAL](https://tools.ietf.org/html/draft-kelly-json-hal-06) because it's simple and lightweight.

```json
{
  "_links": {
    "self": { "href": "/images" },
    "next": { "href": "/images?page=2" }
  },
  "_embedded": {
    "images": []
  },
  "count": 25
}
```

##### HAL Specification

Requirements for a valid HAL document:
- Response Header: `application/hal+json`
- Root object must be a Resource Object with:
  - `_links` - Optional. Contains links to other resources
  - `_embedded` - Optional. Contains embedded resources

#### Response Meta Data

Meta-data about a response goes in a top-level `meta` attribute:

```json
{
  "meta": {
    "representation": "thumbnail"
  },
  "image": {
    "id": "abc123"
  }
}
```

### JSON

Our APIs use JSON because it is the format of choice for most modern Web APIs.

#### Data Formatting

##### Dates and Times

We use [RFC 3339](http://tools.ietf.org/html/rfc3339) in UTC format:

```
2008-09-08T22:47:31Z
```

Format guidelines:
- `T` separates the date from the time
- `Z` indicates UTC
- `YYYY-MM-DD` for dates
- `hh:mm:ss:sss` for timestamps

##### UTF-8

Each API should use UTF-8:

```
Content-Type: application/json; charset=utf-8
```

##### Currency and Fixed-Point Values

```json
{
  "price": {
    "currency_code": "USD",
    "value": 1000,
    "exponent": 2
  }
}
```

### Performance

#### Consumer-Side

The fastest API call is one that isn't made. Consider caching if data can be slightly out of date.

#### Producer-Side

- Use Pagination to manage large result sets
- Use HTTP Compression

##### Condense JSON Response with HTTP Compression

[GZip](http://en.wikipedia.org/wiki/Gzip) compression typically achieves 60-80% reduction in payload size.

##### Conditional GET with Caching and ETag

Include an `ETag` HTTP Header to identify the specific version of the returned resource:

```
Cache-Control: 86400
ETag: "686897696a7c876b7e"
```

| Header | Description |
|--------|-------------|
| `Cache-Control` | Max seconds to cache |
| `ETag` | Version identifier |
| `Last-Modified` | Timestamp of last change |

### Testing

Common test aspects:
- Field validation
- HTTP status codes
- Response structure

#### Improve API Test Performance

Consider:
- Mock Mode: Send back mock objects rather than invoking the service
- Caching: Pull from cache rather than call the service

---

## REST API Reference

The Hybrid Compute REST API provides endpoints for image processing operations. All responses are formatted using [HAL (Hypermedia Application Language)](https://tools.ietf.org/html/draft-kelly-json-hal-06).

See [API Test Results](api/TEST_RESULTS.md) for test execution details.

### Base URL

```
http://localhost:5001/v1
```

### Endpoints

#### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/images` | Upload a new image |
| GET | `/images` | List all uploaded images |
| GET | `/images/:id` | Get image details |
| GET | `/images/:id/file` | Download image file |
| POST | `/images/:id/tiles` | Split image into tiles |

#### Tiles

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tiles` | List all tiles |
| GET | `/tiles/:id` | Get tile details |
| POST | `/tiles/:id/upscale` | Upscale a tile |
| GET | `/tiles/:id/upscaled` | Download upscaled tile |

#### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/stitch` | Stitch tiles into final image |
| GET | `/jobs/:id` | Get job status |

#### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

### Example Requests

**Upload an image:**

```bash
curl -X POST http://localhost:5001/v1/images \
  -F "file=@image.jpg"
```

**List images with pagination:**

```bash
curl "http://localhost:5001/v1/images?offset=0&limit=25"
```

**Response:**
```json
{
  "_links": {
    "self": { "href": "/v1/images?offset=0&limit=25" },
    "next": { "href": "/v1/images?offset=25&limit=25" }
  },
  "_embedded": {
    "images": [
      {
        "id": "abc123",
        "filename": "image.jpg",
        "size": 1024,
        "created_at": "2026-02-19T10:00:00Z"
      }
    ]
  },
  "count": 1,
  "total": 1
}
```

**Create tiles from an image:**

```bash
curl -X POST http://localhost:5001/v1/images/abc123/tiles \
  -H "Content-Type: application/json" \
  -d '{"tile_size": 512}'
```

**Upscale a tile:**

```bash
curl -X POST http://localhost:5001/v1/tiles/tile123/upscale \
  -H "Content-Type: application/json" \
  -d '{"scale": 2}'
```

**Stitch tiles:**

```bash
curl -X POST http://localhost:5001/v1/stitch \
  -H "Content-Type: application/json" \
  -d '{"tile_ids": ["tile1", "tile2"], "rows": 2, "cols": 1}'
```

**Error Response:**

```json
{
  "errors": [
    {
      "code": "not_found",
      "title": "Image not found",
      "details": "No image found with ID: abc123"
    }
  ]
}
```

### Running the API Server

```bash
# Install dependencies
pip install flask werkzeug

# Start the server
python api/server.py
```

The server will start on `http://localhost:5001`

---

## Project Overview

**Hybrid Compute** is a cross-platform GPU-accelerated image processing framework with a CUDA-to-Metal compatibility shim. It enables CUDA-based operations on macOS (Apple Silicon) via Metal, supporting various image processing tasks.

### Architecture

1. **Split**: Process input images locally to create tiles
2. **Transfer**: Transfer tiles to cloud
3. **Upscale**: Upscale tiles on cloud GPU
4. **Stitch**: Stitch upscaled tiles into final image

### Prerequisites

- macOS with Homebrew, Linux (Ubuntu) with apt, or Windows with Chocolatey
- CMake
- OpenCV (for C++ version) or stb_image (for C version)
- NumPy
- Python 3.9+
- Cloud instance with NVIDIA GPU and CUDA toolkit

### Setup

**Quick Setup**

```bash
./scripts/setup.sh
```

**Docker**

```bash
docker build -t hybrid-compute .
docker run --rm hybrid-compute
```

### Usage

**Quick Run**

```bash
./scripts/run.sh
```

**Manual Usage**

1. **Split images into tiles**:
   ```bash
   ./preprocess path/to/input_images/ path/to/tiles/
   ```

2. **Transfer tiles to cloud**:
   ```bash
   export CLOUD_IP="your.cloud.ip"
   ./scripts/transfer_tiles.sh
   ```

3. **Upscale tiles on cloud**:
   ```bash
   cd cloud_gpu && ./upscaler input_tile.jpg output_tile.jpg 2
   ```

4. **Stitch upscaled tiles**:
   ```bash
   python3 scripts/stitch.py path/to/upscaled_tiles/ output_image.jpg
   ```

---

## Git Commit Standards

### Commit Message Format

```
type(scope): short description (≤60 chars)
- optional bullet point 1 (≤72 chars)
- optional bullet point 2
```

### Rules

- **Type**: feat, fix, docs, style, refactor, perf, test, chore, ci, build, revert
- **Scope**: Lowercase with hyphens
- **Description**: Short summary in lowercase (no period)
- **Bullet Points**: Optional, ≤72 characters

### Examples

```
feat(api): add user authentication
- implement jwt token generation

fix(ci): resolve build failures
- update cmake minimum version
```

To enable enforcement:

```bash
cp scripts/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

---

## License

Copyright (c) 2026, bniladridas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice
- Redistributions in binary form must reproduce the above copyright notice in the documentation

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

This software links to [OpenCV](https://opencv.org/license/) and [stb_image](https://github.com/nothings/stb/blob/master/LICENSE).
