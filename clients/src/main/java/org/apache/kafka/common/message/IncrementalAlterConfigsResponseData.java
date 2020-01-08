/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THIS CODE IS AUTOMATICALLY GENERATED.  DO NOT EDIT.

package org.apache.kafka.common.message;

import java.util.ArrayList;
import java.util.List;
import org.apache.kafka.common.protocol.ApiMessage;
import org.apache.kafka.common.protocol.Message;
import org.apache.kafka.common.protocol.MessageUtil;
import org.apache.kafka.common.protocol.Readable;
import org.apache.kafka.common.protocol.Writable;
import org.apache.kafka.common.protocol.types.ArrayOf;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.protocol.types.Schema;
import org.apache.kafka.common.protocol.types.Struct;
import org.apache.kafka.common.protocol.types.Type;


public class IncrementalAlterConfigsResponseData implements ApiMessage {
    private int throttleTimeMs;
    private List<AlterConfigsResourceResult> responses;
    
    public static final Schema SCHEMA_0 =
        new Schema(
            new Field("throttle_time_ms", Type.INT32, "Duration in milliseconds for which the request was throttled due to a quota violation, or zero if the request did not violate any quota."),
            new Field("responses", new ArrayOf(AlterConfigsResourceResult.SCHEMA_0), "The responses for each resource.")
        );
    
    public static final Schema[] SCHEMAS = new Schema[] {
        SCHEMA_0
    };
    
    public IncrementalAlterConfigsResponseData(Readable readable, short version) {
        this.responses = new ArrayList<AlterConfigsResourceResult>();
        read(readable, version);
    }
    
    public IncrementalAlterConfigsResponseData(Struct struct, short version) {
        this.responses = new ArrayList<AlterConfigsResourceResult>();
        fromStruct(struct, version);
    }
    
    public IncrementalAlterConfigsResponseData() {
        this.throttleTimeMs = 0;
        this.responses = new ArrayList<AlterConfigsResourceResult>();
    }
    
    @Override
    public short apiKey() {
        return 44;
    }
    
    @Override
    public short lowestSupportedVersion() {
        return 0;
    }
    
    @Override
    public short highestSupportedVersion() {
        return 0;
    }
    
    @Override
    public void read(Readable readable, short version) {
        this.throttleTimeMs = readable.readInt();
        {
            int arrayLength = readable.readInt();
            if (arrayLength < 0) {
                this.responses = null;
            } else {
                this.responses.clear();
                for (int i = 0; i < arrayLength; i++) {
                    this.responses.add(new AlterConfigsResourceResult(readable, version));
                }
            }
        }
    }
    
    @Override
    public void write(Writable writable, short version) {
        writable.writeInt(throttleTimeMs);
        writable.writeInt(responses.size());
        for (AlterConfigsResourceResult element : responses) {
            element.write(writable, version);
        }
    }
    
    @Override
    public void fromStruct(Struct struct, short version) {
        this.throttleTimeMs = struct.getInt("throttle_time_ms");
        {
            Object[] nestedObjects = struct.getArray("responses");
            this.responses = new ArrayList<AlterConfigsResourceResult>(nestedObjects.length);
            for (Object nestedObject : nestedObjects) {
                this.responses.add(new AlterConfigsResourceResult((Struct) nestedObject, version));
            }
        }
    }
    
    @Override
    public Struct toStruct(short version) {
        Struct struct = new Struct(SCHEMAS[version]);
        struct.set("throttle_time_ms", this.throttleTimeMs);
        {
            Struct[] nestedObjects = new Struct[responses.size()];
            int i = 0;
            for (AlterConfigsResourceResult element : this.responses) {
                nestedObjects[i++] = element.toStruct(version);
            }
            struct.set("responses", (Object[]) nestedObjects);
        }
        return struct;
    }
    
    @Override
    public int size(short version) {
        int size = 0;
        size += 4;
        size += 4;
        for (AlterConfigsResourceResult element : responses) {
            size += element.size(version);
        }
        return size;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof IncrementalAlterConfigsResponseData)) return false;
        IncrementalAlterConfigsResponseData other = (IncrementalAlterConfigsResponseData) obj;
        if (throttleTimeMs != other.throttleTimeMs) return false;
        if (this.responses == null) {
            if (other.responses != null) return false;
        } else {
            if (!this.responses.equals(other.responses)) return false;
        }
        return true;
    }
    
    @Override
    public int hashCode() {
        int hashCode = 0;
        hashCode = 31 * hashCode + throttleTimeMs;
        hashCode = 31 * hashCode + (responses == null ? 0 : responses.hashCode());
        return hashCode;
    }
    
    @Override
    public String toString() {
        return "IncrementalAlterConfigsResponseData("
            + "throttleTimeMs=" + throttleTimeMs
            + ", responses=" + MessageUtil.deepToString(responses.iterator())
            + ")";
    }
    
    public int throttleTimeMs() {
        return this.throttleTimeMs;
    }
    
    public List<AlterConfigsResourceResult> responses() {
        return this.responses;
    }
    
    public IncrementalAlterConfigsResponseData setThrottleTimeMs(int v) {
        this.throttleTimeMs = v;
        return this;
    }
    
    public IncrementalAlterConfigsResponseData setResponses(List<AlterConfigsResourceResult> v) {
        this.responses = v;
        return this;
    }
    
    static public class AlterConfigsResourceResult implements Message {
        private short errorCode;
        private String errorMessage;
        private byte resourceType;
        private String resourceName;
        
        public static final Schema SCHEMA_0 =
            new Schema(
                new Field("error_code", Type.INT16, "The resource error code."),
                new Field("error_message", Type.NULLABLE_STRING, "The resource error message, or null if there was no error."),
                new Field("resource_type", Type.INT8, "The resource type."),
                new Field("resource_name", Type.STRING, "The resource name.")
            );
        
        public static final Schema[] SCHEMAS = new Schema[] {
            SCHEMA_0
        };
        
        public AlterConfigsResourceResult(Readable readable, short version) {
            read(readable, version);
        }
        
        public AlterConfigsResourceResult(Struct struct, short version) {
            fromStruct(struct, version);
        }
        
        public AlterConfigsResourceResult() {
            this.errorCode = (short) 0;
            this.errorMessage = "";
            this.resourceType = (byte) 0;
            this.resourceName = "";
        }
        
        
        @Override
        public short lowestSupportedVersion() {
            return 0;
        }
        
        @Override
        public short highestSupportedVersion() {
            return 0;
        }
        
        @Override
        public void read(Readable readable, short version) {
            this.errorCode = readable.readShort();
            this.errorMessage = readable.readNullableString();
            this.resourceType = readable.readByte();
            this.resourceName = readable.readNullableString();
        }
        
        @Override
        public void write(Writable writable, short version) {
            writable.writeShort(errorCode);
            writable.writeNullableString(errorMessage);
            writable.writeByte(resourceType);
            writable.writeString(resourceName);
        }
        
        @Override
        public void fromStruct(Struct struct, short version) {
            this.errorCode = struct.getShort("error_code");
            this.errorMessage = struct.getString("error_message");
            this.resourceType = struct.getByte("resource_type");
            this.resourceName = struct.getString("resource_name");
        }
        
        @Override
        public Struct toStruct(short version) {
            Struct struct = new Struct(SCHEMAS[version]);
            struct.set("error_code", this.errorCode);
            struct.set("error_message", this.errorMessage);
            struct.set("resource_type", this.resourceType);
            struct.set("resource_name", this.resourceName);
            return struct;
        }
        
        @Override
        public int size(short version) {
            int size = 0;
            size += 2;
            size += 2;
            if (errorMessage != null) {
                size += MessageUtil.serializedUtf8Length(errorMessage);
            }
            size += 1;
            size += 2;
            size += MessageUtil.serializedUtf8Length(resourceName);
            return size;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof AlterConfigsResourceResult)) return false;
            AlterConfigsResourceResult other = (AlterConfigsResourceResult) obj;
            if (errorCode != other.errorCode) return false;
            if (this.errorMessage == null) {
                if (other.errorMessage != null) return false;
            } else {
                if (!this.errorMessage.equals(other.errorMessage)) return false;
            }
            if (resourceType != other.resourceType) return false;
            if (this.resourceName == null) {
                if (other.resourceName != null) return false;
            } else {
                if (!this.resourceName.equals(other.resourceName)) return false;
            }
            return true;
        }
        
        @Override
        public int hashCode() {
            int hashCode = 0;
            hashCode = 31 * hashCode + errorCode;
            hashCode = 31 * hashCode + (errorMessage == null ? 0 : errorMessage.hashCode());
            hashCode = 31 * hashCode + resourceType;
            hashCode = 31 * hashCode + (resourceName == null ? 0 : resourceName.hashCode());
            return hashCode;
        }
        
        @Override
        public String toString() {
            return "AlterConfigsResourceResult("
                + "errorCode=" + errorCode
                + ", errorMessage='" + errorMessage + "'"
                + ", resourceType=" + resourceType
                + ", resourceName='" + resourceName + "'"
                + ")";
        }
        
        public short errorCode() {
            return this.errorCode;
        }
        
        public String errorMessage() {
            return this.errorMessage;
        }
        
        public byte resourceType() {
            return this.resourceType;
        }
        
        public String resourceName() {
            return this.resourceName;
        }
        
        public AlterConfigsResourceResult setErrorCode(short v) {
            this.errorCode = v;
            return this;
        }
        
        public AlterConfigsResourceResult setErrorMessage(String v) {
            this.errorMessage = v;
            return this;
        }
        
        public AlterConfigsResourceResult setResourceType(byte v) {
            this.resourceType = v;
            return this;
        }
        
        public AlterConfigsResourceResult setResourceName(String v) {
            this.resourceName = v;
            return this;
        }
    }
}
